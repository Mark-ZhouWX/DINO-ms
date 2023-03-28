import os
from datetime import datetime

from mindspore.communication import init, get_rank, get_group_size
from torch.utils.tensorboard import SummaryWriter

import mindspore as ms
from mindspore import nn, context, ParallelMode

from common.dataset.dataset import create_mindrecord, create_detr_dataset
from common.detr.matcher.matcher import HungarianMatcher
from common.engine import TrainOneStepWithGradClipLossScaleCell, WithLossCell, set_seed
from config import config
from model_zoo.dino.build_model import build_dino

if __name__ == '__main__':
    # set context, seed
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', pynative_synchronize=True)
    set_seed(0)

    if config.distributed:
        print('distributed training start')
        init(backend_name='nccl')
        rank = get_rank()
        device_num = get_group_size()
        print(f'current rank {rank}/{device_num}')
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=3)

    # create dataset
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=config.num_parallel_workers,
                                  python_multiprocessing=config.python_multiprocessing,
                                  is_training=True)
    ds_size = dataset.get_dataset_size()

    # load pretrained model, only load backbone
    network, criterion = build_dino()
    # pretrain_dir = r"C:\02Data\models" if is_windows else '/home/zhouwuxing/DINO-ms/pretrained_model/'
    # pretrain_path = os.path.join(pretrain_dir, "dino_resnet50_backbone.ckpt")
    pretrain_path = config.pretrain_path
    ms.load_checkpoint(pretrain_path, network, specify_prefix='backbone')
    print(f'successfully load checkpoint from {pretrain_path}')

    epoch_num = 12

    # create optimizer
    lr = 1e-4  # normal learning rate
    lr_backbone = 1e-5  # slower learning rate for pretrained backbone
    lr_drop = epoch_num - 1
    weight_decay = 1e-4
    lr_not_backbone = nn.piecewise_constant_lr(
        [ds_size * lr_drop, ds_size * epoch_num], [lr, lr * 0.1])
    lr_backbone = nn.piecewise_constant_lr(
        [ds_size * lr_drop, ds_size * epoch_num], [lr_backbone, lr_backbone * 0.1])

    backbone_params = list(filter(lambda x: 'backbone' in x.name, network.trainable_params()))
    not_backbone_params = list(filter(lambda x: 'backbone' not in x.name, network.trainable_params()))
    param_dicts = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': not_backbone_params, 'lr': lr_not_backbone, 'weight_decay': weight_decay}
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)

    # # set mix precision
    # dino.to_float(ms.float16)
    # for _, cell in dino.cells_and_names():
    #     if isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, HungarianMatcher)):
    #         cell.to_float(ms.float32)

    # create model with loss scale
    network.set_train(True)
    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    model = WithLossCell(network, criterion)
    model = TrainOneStepWithGradClipLossScaleCell(model, optimizer, scale_sense, grad_clip=True, clip_value=0.1)
    # model = nn.TrainOneStepWithLossScaleCell(dino, optimizer, scale_sense)
    # model = nn.TrainOneStepCell(dino, optimizer)

    # training loop
    log_loss_step = 1
    summary_loss_step = 1
    start_time = datetime.now()
    last_step_time = start_time
    writer = SummaryWriter(f'./work_dirs/tensor_log/{start_time.strftime("%Y_%m_%d_%H_%M_%S")}')
    print(f'prepare finished, entering training loop')
    for e_id in range(epoch_num):
        for s_id, in_data in enumerate(dataset.create_dict_iterator()):
            # image, img_mask(1 for padl), gt_box, gt_label, gt_valid(True for valid)
            input_data = in_data['image'], in_data['mask'], (in_data['labels'], in_data['boxes'], in_data['valid'])
            loss, _, _ = model(*input_data)

            # put on screen
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d - %H:%M:%S")
            past_time = (now - start_time)
            sec_per_step = (now - last_step_time).total_seconds()
            if s_id % log_loss_step == 0:
                print(f"[{now_str}] epoch[{e_id}/{epoch_num}] step[{s_id}/{ds_size}], "
                      f"loss[{loss.asnumpy():.2f}], cost-time[{past_time}], sec-per-step[{sec_per_step:.1f}s]")

            # record in summary for mindinsight
            global_s_id = s_id + e_id * ds_size
            if global_s_id % summary_loss_step == 0:
                writer.add_scalar('loss', loss.asnumpy(), global_s_id)

            last_step_time = now

        # save checkpoint every epoch
        print(f'saving checkpoint for epoch {e_id}')
        ckpt_path = os.path.join(config.output_dir, f'dino_epoch{e_id:03d}_rank{rank}.ckpt')
        ms.save_checkpoint(model, ckpt_path)

    writer.close()
    print(f'finish training for dino')
