import os
import time
from datetime import datetime

from mindspore.communication import init, get_rank, get_group_size
from torch.utils.tensorboard import SummaryWriter

import mindspore as ms
from mindspore import nn, context, set_seed, ParallelMode

from common.dataset.dataset import create_mindrecord, create_detr_dataset
from common.detr.matcher.matcher import HungarianMatcher
from common.engine import TrainOneStepWithGradClipLossScaleCell
from common.utils.system import is_windows
from config import config
from model_zoo.dino.build_model import build_dino

if __name__ == '__main__':
    # set context, seed
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU',
                        pynative_synchronize=False)
    set_seed(0)

    if config.distributed:
        print('distributed training start')
        init()
        rank = get_rank()
        device_num = get_group_size()
        print(f'current rank {rank}/{device_num}')
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        rank = 0
        device_num = 1
    main_device = rank==0

    # create dataset
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=config.num_parallel_workers,
                                  python_multiprocessing=config.python_multiprocessing,
                                  is_training=True)
    ds_size = dataset.get_dataset_size()

    # load pretrained model, only load backbone
    dino = build_dino()
    pretrain_dir = './pretrained_model/'
    pretrain_path = os.path.join(pretrain_dir, "dino_resnet50_backbone.ckpt")
    ms.load_checkpoint(pretrain_path, dino, specify_prefix='backbone')
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

    backbone_params = list(filter(lambda x: 'backbone' in x.name, dino.trainable_params()))
    not_backbone_params = list(filter(lambda x: 'backbone' not in x.name, dino.trainable_params()))
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
    dino.set_train(True)
    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    model = TrainOneStepWithGradClipLossScaleCell(dino, optimizer, scale_sense, grad_clip=True, clip_value=0.1)
    # model = nn.TrainOneStepWithLossScaleCell(dino, optimizer, scale_sense)
    # model = nn.TrainOneStepCell(dino, optimizer)

    # training loop
    os.makedirs(config.output_dir, exist_ok=True)
    log_loss_step = 20
    summary_loss_step = 10
    start_time = last_log_time = datetime.now()
    writer = SummaryWriter(os.path.join(config.output_dir, f'tf_log_{start_time.strftime("%Y_%m_%d_%H_%M_%S")}'))
    for e_id in range(epoch_num):
        for s_id, in_data in enumerate(dataset.create_dict_iterator()):
            global_s_id = s_id + e_id * ds_size
            # image, img_mask(1 for padl), gt_box, gt_label, gt_valid(True for valid)
            loss, _, _ = model(in_data['image'], in_data['mask'], in_data['boxes'], in_data['labels'], in_data['valid'])

            # put on screen
            now = datetime.now().strftime("%Y-%m-%d - %H:%M:%S")
            past_time = (datetime.now() - start_time)
            if main_device:
                if global_s_id % log_loss_step == 0:
                    step_time = datetime.now() - last_log_time
                    step_time_s = step_time.total_seconds() / log_loss_step

                    past_time_min, past_time_sec = divmod(past_time.seconds, 60)
                    past_time_hour, past_time_min = divmod(past_time_min, 60)

                    rema_step_time_s = (epoch_num * ds_size - global_s_id) * step_time_s
                    rema_time_min, rema_time_sec = divmod(int(rema_step_time_s), 60)
                    rema_time_hour, rema_time_min = divmod(rema_time_min, 60)
                    rema_time_day, rema_time_hour = divmod(rema_time_hour, 24)

                    print(f"[{now}] epoch[{e_id+1}/{epoch_num}] step[{s_id}/{ds_size}], "
                          f"loss[{loss.asnumpy():.2f}] "
                          f"past-t[{past_time.days:02d}d {past_time_hour:02d}:{past_time_min:02d}:{past_time_sec:02d}] "
                          f"rema-t[{rema_time_day:02d}d {rema_time_hour:02d}:{rema_time_min:02d}:{rema_time_sec:02d}] "
                          f"step-t[{step_time_s:.1f}s]")
                    last_log_time = datetime.now()

                # record in summary for mindinsight
                if global_s_id % summary_loss_step == 0:
                    writer.add_scalar('loss', loss.asnumpy(), global_s_id)
                    writer.flush()
        if main_device:
            # save checkpoint every epoch
            ckpt_path = os.path.join(config.output_dir, f'dino_epoch{e_id+1:03d}.ckpt')
            print(f'saving checkpoint for epoch {e_id + 1} at {ckpt_path}')
            ms.save_checkpoint(dino, ckpt_path)

    writer.close()
    print(f'finish training for dino')
