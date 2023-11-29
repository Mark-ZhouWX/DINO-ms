# DINO

This repository is an implementation of _[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)_ with MindSpore.
Curently only Ascend pynative is supported.

# Prepare Dataset and Pretrained Model
Download [COCO2017 dataset](https://cocodataset.org/#download) and put it under directory `${project_root}/data/`.

Download pretrained model([ResNet50](https://download.mindspore.cn/toolkits/minddetr/dino/ms_dino_r50_4scale_12ep_49_2AP.ckpt)), and put it under directory `${project_root}/pretrained_model/`.


# Evaluation
Run
```shell
cd ${project_root}
python eval.py
```


# Train
Run
```shell
cd ${project_root}
python train.py
```
