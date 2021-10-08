# Action Recognition Study

This repository contains a general implementation of 6 representative 2D and 3D approaches for action recognition including I3D [1], ResNet3D [2], S3D [3], R(2+1)D [4], TSN [5] and TAM [6].
And the codes are used for our analysis on action recognition.

Chun-Fu (Richard) Chen*, Rameswar Panda*, Kandan Ramakrishnan, Rogerio Feris, John Cohn, Aude Oliva and Quanfu Fan*, "Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition".  
*: Equal contributions

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{chen2020deep,
    title={Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition},
    author={Chen, Chun-Fu and Panda, Rameswar and Ramakrishnan, Kandan and Feris, Rogerio and Cohn, John and Oliva, Aude and Fan, Quanfu},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021},
    month = jun
}
```


## Requirements
```
pip install -r requirement.txt
```

## Data Preparation
The dataloader (utils/video_dataset.py) can load videos (image sequences) stored in the following format:
```
-- dataset_dir
---- train.txt
---- val.txt
---- test.txt
---- videos
------ video_0_folder
-------- 00001.jpg
-------- 00002.jpg
-------- ...
------ video_1_folder
------ ...
```

Each line in `train.txt` and `val.txt` includes 4 elements and separated by a symbol, e.g. space or semicolon. 
Four elements (in order) include (1)relative paths to `video_x_folder` from `dataset_dir`, (2) starting frame number, usually 1, (3) ending frame number, (4) label id (a numeric number).

E.g., a `video_x` has `300` frames and belong to label `1`.
```
path/to/video_x_folder 1 300 1
```
The difference for `test.txt` is that each line will only have 3 elements (no label information).

After that, you need to update the `utils/data_config.py` for the datasets accordingly.

We provided three scripts in the `tools` folder to help convert some datasets but the details in the scripts must be set accordingly. E.g., the path to videos.

## Mini-datasets
Please find [mini_ssv2](mini_ssv2.txt) and [mini_kinetics400](mini_kinetics400.txt) for the used classes.

## Training and Evaluation
The `opts.py` illustrates the available options for training 2D and 3D models. Some options are only for 2D models or 3D models.

You can get help via
```
python3 train.py --help
```


Here is an example to train a `64-frame I3D` on the `Kinetics400` datasets with `Uniform Sampling` as input.

```
python3 train.py --datadir /path/to/folder --threed_data \
--dataset kinetics400 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d -b 64 -j 64
```

To evaluate a model with more crops and clips, we provided `test.py` to support those variants. 
The `test.py` shares the same `opts.py` to `train.py`, so most of settings are the same.
Furthermore, you can set `num_crops` and `num_clips` for `test.py`.

Here is an example to evaluate on the above model with 3 crops and 3 clips

```
python3 test.py --datadir /path/to/folder --threed_data \
--dataset kinetics400 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d -b 64 -j 64 \
-e --pretrained /path/to/file --num_clips 3 --num_crops 3
```


## Models and Results

We provided some pretrained models with `32` frames as input without temporal pooling. 
Those models can be evaluated with following command template, and appending additional configs.

Note: you might need to change batch size based on your GPU memory.

Models can be directly downloaded from the [asset](https://github.com/IBM/action-recognition-pytorch/releases/tag/weights-v0.1).

### Kinetics400
```bash
python3 test.py --groups 32 -e --frames_per_group 2 --without_t_stride --logdir logs/ --dataset kinetics400 \
--num_crops 3 --num_clips 10 --input_size 256 --disable_scaleup -b 6 -j 24 --dense_sampling \
--datadir /path/to/dataset \
--pretrained /path/to/pretrained_model
```

| Model | Top-1 Acc | Additional configs | 
|-------|-----------| --- |
| [I3D-ResNet-50-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/K400-I3D-ResNet-50-f32.pth.tar) | 76.61% | --backbone_net i3d_resnet -d 50 | 
| [TAM-ResNet-50-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/K400-TAM-ResNet-50-f32.pth.tar) | 76.18% | --backbone_net resnet -d 50 --temporal_module_name TAM |
| [I3D-ResNet-101-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/K400-I3D-ResNet-101-f32.pth.tar) | 77.80% | --backbone_net i3d_resnet -d 101 |
| [TAM-ResNet-101-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/K400-TAM-ResNet-101-f32.pth.tar) | 77.61% | --backbone_net resnet -d 101 --temporal_module_name TAM |

### Something-Something-V2
```bash
python3 test.py --groups 32 -e --frames_per_group 1 --without_t_stride --logdir logs/ --dataset st2stv2 \
--num_crops 3 --num_clips 2 --input_size 256 --disable_scaleup -b 6 -j 24  \
--datadir /path/to/dataset \
--pretrained /path/to/pretrained_model
```

| Model | Top-1 Acc | Additional configs |
|-------|-----------|--|
| [I3D-ResNet-50-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-I3D-ResNet-50-f32.pth.tar) | 62.84% | --backbone_net i3d_resnet -d 50 |
| [TAM-ResNet-50-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-TAM-ResNet-50-f32.pth.tar) | 63.83% | --backbone_net resnet -d 50 --temporal_module_name TAM |
| [I3D-ResNet-101-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-I3D-ResNet-101-f32.pth.tar) | 64.29% | --backbone_net i3d_resnet -d 101 |
| [TAM-ResNet-101-f32](https://github.com/IBM/action-recognition-pytorch/releases/download/weights-v0.1/SSV2-TAM-ResNet-101-f32.pth.tar) | 65.32% | --backbone_net resnet -d 101 --temporal_module_name TAM |


### Results on Mini-Datasets
See [benchmark_mini.md](./benchmark_mini.md)



## Reference

1. Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017

2. Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs
and ImageNet? In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

3. Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking Spatiotemporal Feature Learning:
Speed-Accuracy Trade-offs in Video Classification. In The European Conference on Computer Vision (ECCV),
Sept. 2018

4. Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A Closer Look at Spatiotemporal
Convolutions for Action Recognition. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018

5. Ji Lin, Chuang Gan, and Song Han. Temporal Shift Module for Efficient Video Understanding. In The IEEE International
Conference on Computer Vision (ICCV), 2019

6. Quanfu Fan, Chun-Fu (Ricarhd) Chen, Hilde Kuehne, Marco Pistoia, and David Cox. More Is Less: Learning Efficient
Video Representations by Temporal Aggregation Modules. In Advances in Neural Information Processing Systems 33,
2019.
