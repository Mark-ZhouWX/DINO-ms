import os

import torch
import platform

from test import is_windows


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    # print(par_dict)
    for name, value in par_dict['model'].items():
        print(name, value.numpy().shape)
        pt_params[name] = value.numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(name, value.shape)
        ms_params[name] = value
    return ms_params


if __name__ == "__main__":
    from test.dino import dino
    pth_dir = r"C:\02Data\models" if is_windows else '/data/zhouwuxing/pretrained_model/'
    pth_path = os.path.join(pth_dir, "dino_r50_4scale_12ep_49_2AP.pth")

    pt_param = pytorch_params(pth_path)
    print("=" * 20)
    ms_param = mindspore_params(dino)
