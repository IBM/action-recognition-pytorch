# Action Recognition Study

This repository contains a general implementation of 6 representative 2D and 3D approaches for action recognition including I3D [1], ResNet3D [2], S3D [3], R(2+1)D [4], TSN [5] and TAM [6]. 

Results of mini-datasets and full-datasets can be found at [mini-datasets](./benchmark_mini.md) and [full-datasets](./benchmark_full.md). The trained models will be released soon.

```
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
