from datetime import datetime

from mindspore import Model, nn, LossMonitor, context, set_seed, ops
from mindspore import dataset as ds
from mindspore.nn import WithGradCell
from mindspore.ops import composite, functional

from common.dataset.dataset import create_mindrecord, create_detr_dataset
from test.dino import dino
from config import config


if __name__ == '__main__':
    # set context, seed
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    set_seed(0)
    rank = 0
    device_num = 1

    # create dataset
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=config.num_parallel_workers,
                                  python_multiprocessing=config.python_multiprocessing,
                                  is_training=True)

    # create optimizer
    optimizer = nn.AdamWeightDecay(dino.trainable_params(), learning_rate=1e-3)

    # create model with loss scale
    dino.set_train(True)
    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    # model = nn.TrainOneStepWithLossScaleCell(dino, optimizer, scale_sense)
    model = nn.TrainOneStepCell(dino, optimizer)

    # load pretrained model
    pass

    # training loop
    epoch_num = 3
    log_loss_step = 1
    for e_id in range(epoch_num):
        ds_size = dataset.get_dataset_size()
        for s_id, in_data in enumerate(dataset.create_dict_iterator()):
            # image, img_mask(1 for padl), gt_box, gt_label, gt_valid(True for valid)
            loss = model(in_data['image'], in_data['mask'], in_data['boxes'], in_data['labels'], in_data['valid'])
            now = datetime.now().strftime("%Y-%m-%d - %H:%M:%S")
            if s_id % log_loss_step == 0:
                print(f"time[{now}] epoch[{e_id}/{epoch_num}] step[{s_id}/{ds_size}], "
                      f"loss[{loss.asnumpy()}]")
    print(f'finish training for dino')
