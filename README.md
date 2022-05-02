# MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Guanxiong Sun](https://sunguanxiong.github.io).

This repo is an self implemention of ["MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection"](https://ojs.aaai.org/index.php/AAAI/article/view/16365/16172) in AAAI 2021. This repository contains a PyTorch implementation of the approach MAMBA based on [MEGA](https://github.com/Scalsol/mega.pytorch), as well as some training scripts to reproduce the results on the ImageNet VID dataset.

Thanks to MEGA's codebase, this repository also implements several other algorithms like [FGFA](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html), [SELSA](https://arxiv.org/abs/1907.06390), and [RDN](https://arxiv.org/abs/1908.09511). We hope this repository would help further research in the field of video object detection and beyond. :)

## Citing MAMBA

Please cite our paper in your publications if it helps your research:

```
@inproceedings{sun2021mamba,
  title={MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection},
  author={Sun, Guanxiong and Hua, Yang and Hu, Guosheng and Robertson, Neil},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2620--2627},
  year={2021}
}
```

## Main Results

Pretrained models are now available at [Baidu](https://pan.baidu.com/s/1qjIAD3ohaJO8EF1mZ4nLEg) (code: neck) and Google Drive.

|         Model         |  Backbone  | AP50 | AP (fast) | AP (med) | AP (slow) |                                             Link                                             |
| :-------------------: | :--------: | :--: | :-------: | :------: | :-------: | :------------------------------------------------------------------------------------------: |
| single frame baseline | ResNet-101 | 76.7 |   52.3    |   74.1   |   84.9    | [Google](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing) |
|          DFF          | ResNet-101 | 75.0 |   48.3    |   73.5   |   84.5    | [Google](https://drive.google.com/file/d/1Dn_RQRlA7z2XkRRS4XERUW_UH9jlNvMo/view?usp=sharing) |
|         FGFA          | ResNet-101 | 78.0 |   55.3    |   76.9   |   85.6    | [Google](https://drive.google.com/file/d/1yVgy7_ff1xVD1SooqbcK-OzKMgPpUcg4/view?usp=sharing) |
|       RDN-base        | ResNet-101 | 81.1 |   60.2    |   79.4   |   87.7    | [Google](https://drive.google.com/file/d/1jM5LqlVtCGjKH-MocTCjzFIVjqCyng8M/view?usp=sharing) |
|          RDN          | ResNet-101 | 81.7 |   59.5    |   80.0   |   89.0    | [Google](https://drive.google.com/file/d/1FgoOwj-GFAMVn2hkSFKnxn5fKWPSxlUF/view?usp=sharing) |
|         MEGA          | ResNet-101 | 82.9 |   62.7    |   81.6   |   89.4    | [Google](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing) |
|       **MAMBA_INS**   | ResNet-101 | 83.5 |   66.3    |   82.8   |   88.9    |                                          model url                                           |

**Note**: The motion-IoU specific AP evaluation code is a bit different from the original implementation in FGFA. I think the original implementation is really weird so I modify it. So the results may not be directly comparable with the results provided in FGFA and other methods that use MXNet version evaluation code. But we could tell which method is relatively better under the same evaluation protocol.

## Installation

### Requirements:

- Python 3.7
- PyTorch 1.5
- torchvision 0.6
- GCC = 7.5.0
- OpenCV = 4.5.5
- CUDA = 10.2

### Environment and compilation

```bash
# install PyTorch 1.5 with CUDA 10.2
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch

# install other dependencies
pip install -r requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex
cd apex
# reset to an old version, bug reported at https://github.com/NVIDIA/apex/issues/1043
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# complilation
cd ..
python setup.py build develop
```

## Data preparation

### Option 1: download from external address

Please download [ILSVRC2015 DET](https://image-net.org/data/ILSVRC/2015/ILSVRC2015_DET.tar.gz) and [ILSVRC2015 VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz) dataset. More information of the datasets can be found [here](https://image-net.org/challenges/LSVRC/2015/2015-downloads). After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./datasets/ILSVRC2015/
    ./datasets/ILSVRC2015/Annotations/DET
    ./datasets/ILSVRC2015/Annotations/VID
    ./datasets/ILSVRC2015/Data/DET
    ./datasets/ILSVRC2015/Data/VID
    ./datasets/ILSVRC2015/ImageSets

You can symlink the data to the dataset folder, so the path should be `datasets/ILSVRC2015`.

**Note**: We have already provided a list of all images we use to train and test our model as txt files under directory `datasets/ILSVRC2015/ImageSets`. You do not need to change them.

## Usage

**Note**: Cache files will be created at the first time you run this project, this may take some time! Don't worry!

**Note**: Currently, one GPU could only hold 1 image. Do not put 2 or more images on 1 GPU!

**Note** We provide template files named `BASE_RCNN_{}gpus.yaml` which would automatically change the batch size and other relevant settings. This behavior is similar to detectron2. If you want to train model with different number of gpus, please change it by yourself :) But assure **1 GPU only holds 1 image**! That is to say, you should always keep `SOLVER.IMS_PER_BATCH` and `TEST.IMS_PER_BATCH` equal to the number of GPUs you use.

### Training

The following command line will train MAMBA_INS_R_101_C4_1x on 4 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        tools/train_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
        --motion-specific \
        OUTPUT_DIR training_dir/MAMBA/vid_R_101_C4_INS_1x

Please note that: 0) `nproc_per_node` is equal to the number of GPUs. If you are using 1 GPU, you should modify it.
Besides, during training and testing, there should be 1 image per GPU. For example, if you are using 1 GPU, the command
for inference should be

```
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
    --motion-specific \
    OUTPUT_DIR training_dir/MAMBA/vid_R_101_C4_INS_1x SOLVER.IMS_PER_BATCH 1
```

1. The models will be saved into `OUTPUT_DIR`.
2. If you want to train MAMBA and other methods with other backbones, please change `--config-file`.
3. For training FGFA and DFF, we need pretrained weight of FlowNet. We provide the converted version [here](https://drive.google.com/file/d/1gib7XtS1fSYDTM9RnUJ72a3vREV_6SJH/view?usp=sharing). After downloaded, it should be placed at `models/`. See `config/defaults.py` and the code for further details.
4. For training RDN, we adopt the same two-stage training strategy as described in its original paper. The first phase should be run with config file `configs/RDN/vid_R_101_C4_RDN_base_1x.yaml`. For the second phase, `MODEL.WEIGHT` should be set to the filename of the final model of the first stage training. Or you could rename the model's filename to `RDN_base_R_101.pth` and put it under `models/` and directly train the second phase with config file `configs/RDN/vid_R_101_C4_RDN_1x.yaml`.
5. If you do not want to evaluate motion-IoU specific AP at the end of training, simply deleting `--motion-specific`.

### Inference

The inference command line for testing on the validation dataset:

    python -m torch.distributed.launch \
        --nproc_per_node 4 \
        tools/test_net.py \
        --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
        --motion-specific \
        MODEL.WEIGHT MAMBA_INS_R_101.pth

Please note that: 0) `nproc_per_node` is equal to the number of GPUs. If you are using 1 GPU, you should modify it.
Besides, during training and testing, there should be 1 image per GPU. For example, if you are using 1 GPU, the command
for inference should be

```
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    tools/test_net.py \
    --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
    --motion-specific \
    MODEL.WEIGHT MAMBA_INS_R_101.pth  TEST.IMS_PER_BATCH 1
```

1. If your model's name is different, please replace `MAMBA_INS_R_101.pth` with your own.
2. If you want to evaluate a different model, please change `--config-file` to its config file and `MODEL.WEIGHT` to its weights file.
3. If you do not want to evaluate motion-IoU specific AP, simply deleting `--motion-specific`.
4. Testing is time-consuming, so be patient!
5. As testing on above 170000+ frames is toooo time-consuming, so we enable directly testing on generated bounding boxes, which is automatically saved in a file named `predictions.pth` on your training directory. That means you do not need to run the evaluation from the very start every time. You could access this by running:

```
    python tools/test_prediction.py \
        --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
        --prediction [YOUR predictions.pth generated by MAMBA]
        --motion-specific
```

### Visualisation of Results
To visualise the results during inference, the config option `TEST.DEBUG` 
controls the process of visualisation.

There are several available options.
```
TEST.DEBUG.LEVEL=0 (default), no visualisation.
TEST.DEBUG.LEVEL=1, save visualisation results onto the disk.
TEST.DEBUG.LEVEL=2, visualise using opencv window as run.
```
**Note** When `TEST.DEBUG.LEVEL=2`, you may want to pause the visualisation results
by setting `TEST.DEBUG.PAUSE=True`.

For example, if you want to save all visualisation results using 1 GPU, the command should be

```
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    tools/test_net.py \
    --config-file configs/MAMBA/vid_R_101_C4_INS_1x.yaml \
    --motion-specific \
    MODEL.WEIGHT MAMBA_INS_R_101.pth \
    TEST.IMS_PER_BATCH 1 \
    TEST.DEBUG.LEVEL=1
```
The visualisation code is referenced from [CenterNet](https://github.com/xingyizhou/CenterNet).

### Customize

If you want to use these methods on your own dataset or implement your new method. Please follow [CUSTOMIZE.md](CUSTOMIZE.md).
