# DINO-ms

DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

![network](https://github.com/IDEA-Research/detrex/blob/main/projects/dino/assets/dino_arch.png)

## Installation

The code requires `python>=3.7` and `mindspore>=2.1` and currently supports GPU platform. Please follow the instructions [here](https://www.mindspore.cn/install) to install mindspore dependencies.

Clone the repository locally and install with

```shell
git clone https://github.com/Mark-ZhouWX/models.git
pip install -r requirements.txt
```

## Train
First put your dataset under {project_root}/datasets (currently only COCO is supported), and download pretrained model from [here](https://download.mindspore.cn/toolkits/minddetr/dino/dino_resnet_backbone-e295598d.ckpt), then run:
```shell
python train.py
```
for single card training, or
```shell
mpirun --allow-run-as-root -n 8 python train.py
```
for multi-card training

## Inference
For inference, please run:
```shell
python eval.py --eval_model_path /path/to/model
```

