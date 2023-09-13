import os
import random
import re

import cv2
import mindspore
import torch

from test.dino import build_dino, convert_input_format, get_input

dino = build_dino()

# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file, verbose=False):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    # print(par_dict)
    for name, value in par_dict['model'].items():
        if verbose:
            print(name, value.numpy().shape)
        pt_params[name] = value.numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network, verbose=False):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        if verbose:
            print(name, value.shape)
        ms_params[name] = value
    return ms_params


def map_resnet(ms_name):

    # first layer
    if ms_name == 'backbone.conv1.weight':
        return 'backbone.stem.conv1.weight'
    if ms_name == 'backbone.bn1.moving_mean':
        return 'backbone.stem.conv1.norm.running_mean'
    if ms_name == 'backbone.bn1.moving_variance':
        return 'backbone.stem.conv1.norm.running_var'
    if ms_name == 'backbone.bn1.gamma':
        return 'backbone.stem.conv1.norm.weight'
    if ms_name == 'backbone.bn1.beta':
        return 'backbone.stem.conv1.norm.bias'

    pattern_conv_weight = r'backbone.layer(\d).(\d).conv(\d).weight'
    res = re.match(pattern_conv_weight, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.conv{c_id}.weight'

    pattern_bn_mean = r'backbone.layer(\d).(\d).bn(\d).moving_mean'
    res = re.match(pattern_bn_mean, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.conv{c_id}.norm.running_mean'

    pattern_bn_variance = r'backbone.layer(\d).(\d).bn(\d).moving_variance'
    res = re.match(pattern_bn_variance, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.conv{c_id}.norm.running_var'

    pattern_bn_gamma = r'backbone.layer(\d).(\d).bn(\d).gamma'
    res = re.match(pattern_bn_gamma, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.conv{c_id}.norm.weight'

    pattern_bn_beta = r'backbone.layer(\d).(\d).bn(\d).beta'
    res = re.match(pattern_bn_beta, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.conv{c_id}.norm.bias'

    # shortcut
    pattern_downsample_conv_weight = r'backbone.layer(\d).(\d).down_sample.(\d).weight'
    res = re.match(pattern_downsample_conv_weight, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.shortcut.weight'

    pattern_downsample_bn_mean = r'backbone.layer(\d).(\d).down_sample.(\d).moving_mean'
    res = re.match(pattern_downsample_bn_mean, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.shortcut.norm.running_mean'

    pattern_downsample_bn_variance = r'backbone.layer(\d).(\d).down_sample.(\d).moving_variance'
    res = re.match(pattern_downsample_bn_variance, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.shortcut.norm.running_var'

    pattern_downsample_bn_gamma = r'backbone.layer(\d).(\d).down_sample.(\d).gamma'
    res = re.match(pattern_downsample_bn_gamma, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.shortcut.norm.weight'

    pattern_downsample_bn_beta = r'backbone.layer(\d).(\d).down_sample.(\d).beta'
    res = re.match(pattern_downsample_bn_beta, ms_name, flags=0)
    if res is not None:
        l_id = res.groups()[0]
        b_id = res.groups()[1]
        c_id = res.groups()[2]
        return f'backbone.res{int(l_id)+1}.{b_id}.shortcut.norm.bias'
    raise ValueError(ms_name)


def mapper(ms_name: str):
    # resnet backbone
    if ms_name.startswith('backbone.'):
        return map_resnet(ms_name)

    if 'gamma' in ms_name:
        return ms_name.replace('gamma', 'weight')
    if 'beta' in ms_name:
        return ms_name.replace('beta', 'bias')
    if 'embedding_table' in ms_name:
        return ms_name.replace('embedding_table', 'weight')

    if 'out_proj' in ms_name:
        return ms_name.replace('out_proj', 'attn.out_proj')

    for n in ['in_projs.0.weight', 'in_projs.1.weight', 'in_projs.2.weight']:
        if n in ms_name:
            return ms_name.replace(n, 'attn.in_proj_weight')
    for n in ['in_projs.0.bias', 'in_projs.1.bias', 'in_projs.2.bias']:
        if n in ms_name:
            return ms_name.replace(n, 'attn.in_proj_bias')

    return ms_name


def map_torch_to_mindspore(ms_dict, torch_dict, verbose=False, backbone_only=False):
    new_params_list = []
    for name, value in ms_dict.items():
        if backbone_only and not name.startswith('backbone.'):
            continue
        torch_name = mapper(name)
        torch_value = torch_dict[mapper(name)]

        convert_value = mindspore.Tensor(torch_value)
        for i, n in enumerate(['in_projs.0.weight', 'in_projs.1.weight', 'in_projs.2.weight',
                               'in_projs.0.bias', 'in_projs.1.bias', 'in_projs.2.bias']):
            if n in name:
                convert_value = convert_value[i % 3 * 256: (i % 3 + 1) * 256]
                break
        if verbose:
            print(name, value.shape)
            # print(torch_name, value.shape)
        assert value.shape == convert_value.shape, f"value shape not match, ms {name} {value.shape}, torch {torch_name}{convert_value.shape}"
        new_params_list.append(dict(name=name, data=convert_value))
    return new_params_list


def convert_parameter(i_pth_path, i_ms_pth_path, backbone_only=False):
    pt_param = pytorch_params(i_pth_path)
    # print('\n'*5)
    ms_param = mindspore_params(dino)
    ms_params_list = map_torch_to_mindspore(ms_param, pt_param, verbose=False, backbone_only=backbone_only)

    print(f'successfully convert the checkpoint, saved as {i_ms_pth_path}')
    mindspore.save_checkpoint(ms_params_list, i_ms_pth_path)
    mindspore.load_checkpoint(i_ms_pth_path, dino)
    print(f'successfully load checkpoint into dino network')

    # compare transformer class and outer class
    if backbone_only:
        print(f'torch conv1', pt_param['backbone.stem.conv1.weight'][0, 0, 0, :5])
        print(f'ms conv1', dino.backbone.conv1.weight[0, 0, 0, :5])
    else:
        l_id = 6
        print(f'torch trans class {l_id}', pt_param[f'transformer.decoder.class_embed.{l_id}.weight'][0, :5])
        print(f'torch class {l_id}', pt_param[f'class_embed.{l_id}.weight'][0, :5])

        print(f'ms trans class 0', dino.transformer.decoder.class_embed[l_id].weight[0, :5])
        print(f'ms class', dino.class_embed[l_id].weight[0, :5])


def che_res(in_pth_path, in_ms_pth_path):
    dino.set_train(False)
    mindspore.load_checkpoint(in_ms_pth_path, dino)

    # test load
    pt_param = pytorch_params(in_pth_path)
    b_id = 2
    print(f"torch backbone res2.1.conv1.norm.running_mean", pt_param[f'backbone.res2.{b_id}.conv1.norm.running_mean'][:5])
    print(f"ms backbone res2.1.bn1.moving_mean", dino.backbone.layer1[b_id].bn1.moving_mean[:5])
    print(f"torch stem",
          pt_param[f'backbone.stem.conv1.weight'][4,2,3,:])  # 64,3,7,7
    print(f"ms backbone stem", dino.backbone.conv1.weight[4,2,3,:])

    l_id = 6
    print(f'torch trans class {l_id}', pt_param[f'transformer.decoder.class_embed.{l_id}.weight'][0, :5])
    print(f'ms trans class 0', dino.transformer.decoder.class_embed[l_id].weight[0, :5])
    print(f'ms class', dino.class_embed[l_id].weight[0, :5])

    inputs, _ = get_input()
    images, img_masks, gt_classes_list, gt_boxes_list, gt_valids_list = convert_input_format(inputs)
    inputs = images, img_masks, gt_boxes_list, gt_classes_list, gt_valids_list
    infer_result = dino(inputs)
    print('\n'*3, 'infer result')
    print('scores', infer_result[0]['instances']['scores'][:5])
    print('pred_boxes\n', infer_result[0]['instances']['pred_boxes'][:5])
    print('pred_classes', infer_result[0]['instances']['pred_classes'][:5])

    batch_id = 0
    bbox_id = 0
    img = inputs[batch_id]['image'].transpose(1, 2, 0).asnumpy()
    bboxes = infer_result[batch_id]['instances']['pred_boxes']
    for bbox_id in range(5):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (int(bboxes[bbox_id,0]), int(bboxes[bbox_id,1])), (int(bboxes[bbox_id,2]), int(bboxes[bbox_id,3])), color, 2)

    cv2.imshow('soccer', img)
    cv2.waitKey()


if __name__ == "__main__":

    pth_dir = './pretrained_model/'
    pth_path = os.path.join(pth_dir, "torch_resnet_backbone.pth")
    ms_pth_path = os.path.join(pth_dir, "ms_torch_like_init.ckpt")

    convert_parameter(pth_path, ms_pth_path)
    # che_res(pth_path, ms_pth_path)


