## Experimental Setting
- All trained from 8-frame model (finetune from ImageNet-pretrained), while more frame models are finetuned from fewer-frame model. 
- Number of frames (`-fX`, `X` denotes frame number.)
- Backbone (2D or 3D)
- With or without temporal pooling in the model (`-tp` means with temporal pooling)


## Performance

The model name should self-explain the experimental setting

### Dataset:  Mini-Something-Something-V2

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs | 
|------------|-------|-------|-------|--------|--------|
| mini_st2stv2-inception-v1-f8-cosine-bs72-e75 | 33.1384 | 63.6156 | 11,978,817,536 | 5,689,079 | 75|
| mini_st2stv2-inception-v1-f16-multisteps-bs72-e35 | 34.6555 | 65.2852 | 23,957,635,072 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-f32-multisteps-bs48-e35 | 34.5199 | 66.3022 | 47,915,270,144 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-f64-multisteps-syncbn-bs36-e35 | 36.3228 | 67.4119 | 95,830,540,288 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-tp-f8-cosine-bs72-e75 | 53.2672 | 81.0153 | 6,521,725,952 | 5,689,079 | 75|
| mini_st2stv2-inception-v1-tp-f16-multisteps-bs72-e35 | 54.5555 | 82.9138 | 13,043,451,904 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-tp-f32-multisteps-bs72-e35 | 55.1064 | 83.7105 | 26,086,903,808 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e35 | 57.4187 | 85.6877 | 52,173,807,616 | 5,689,079 | 35|
| | | | | |
| mini_st2stv2-resnet-18-f8-cosine-bs72-e75 | 29.5534 | 59.4118 | 14,508,490,752 | 11,221,143 | 75|
| mini_st2stv2-resnet-18-f16-multisteps-bs72-e35 | 30.9263 | 61.5815 | 29,016,981,504 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-f32-multisteps-bs72-e35 | 30.9348 | 61.2255 | 58,033,963,008 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-f64-multisteps-syncbn-bs36-e35 | 32.08 | 62.9573 | 116,067,926,016 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-tp-f8-cosine-bs72-e75 | 50.0636 | 78.7016 | 7,520,780,288 | 11,221,143 | 75|
| mini_st2stv2-resnet-18-tp-f16-multisteps-bs72-e35 | 51.6145 | 81.473 | 15,041,560,576 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-tp-f32-multisteps-bs72-e35 | 52.4451 | 81.9561 | 30,083,121,152 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-tp-f64-multisteps-bs48-e35 | 54.7928 | 83.8546 | 60,166,242,304 | 11,221,143 | 35|
| | | | | |
| mini_st2stv2-resnet-50-f8-cosine-bs72-e75 | 33.9351 | 63.9885 | 32,697,090,048 | 23,686,295 | 75|
| mini_st2stv2-resnet-50-f16-multisteps-bs60-e35 | 35.342 | 66.8785 | 65,394,180,096 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-f32-multisteps-syncbn-bs36-e35 | 36.2551 | 67.6575 | 130,788,360,192 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-f64-multisteps-syncbn-bs36-e35 | 36.7971 | 68.1233 | 261,576,720,384 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-tp-f8-cosine-bs72-e75 | 53.4961 | 80.2526 | 14,135,984,128 | 23,686,295 | 75|
| mini_st2stv2-resnet-50-tp-f16-multisteps-bs72-e35 | 55.8946 | 83.4901 | 28,271,968,256 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e35 | 56.9225 | 84.5873 | 56,543,936,512 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e35 | 59.3157 | 86.8903 | 113,087,873,024 | 23,686,295 | 35|
| | | | | |
| | | | | |
| mini_st2stv2-s3d-f8-cosine-bs72-e75 | 57.115 | 84.2105 | 19,778,626,560 | 7,992,503 | 75|
| mini_st2stv2-s3d-f16-multisteps-bs72-e35 | 62.7426 | 87.7617 | 39,712,547,840 | 7,992,503 | 35|
| mini_st2stv2-s3d-f32-multisteps-syncbn-bs36-e35 | 65.0576 | 89.2446 | 79,580,390,400 | 7,992,503 | 35|
| mini_st2stv2-s3d-f64-multisteps-syncbn-bs36-e35 | 67.8438 | 91.1331 | 159,316,075,520 | 7,992,503 | 35|
| mini_st2stv2-s3d-tp-f8-cosine-bs72-e75 | 50.8941 | 78.8287 | 12,264,670,208 | 7,992,503 | 75|
| mini_st2stv2-s3d-tp-f16-multisteps-bs72-e35 | 57.115 | 84.5156 | 24,529,340,416 | 7,992,503 | 35|
| mini_st2stv2-s3d-tp-max-f32-multisteps-bs60-e35| 61.590 | 87.8380 | 49,058,680,832 | 7,992,503 | 35|
| mini_st2stv2-s3d-tp-max-f64-multisteps-syncbn-bs36-e35 | 67.2172 | 91.0993 | 98,117,361,664 | 7,992,503 | 35|
| | | | | |
| mini_st2stv2-s3d-resnet-18-f8-cosine-bs72-e75 | 57.4286 | 83.8715 | 21,329,215,488 | 15,425,559 | 75|
| mini_st2stv2-s3d-resnet-18-f16-multisteps-bs72-e35 | 63.5054 | 88.6007 | 42,658,430,976 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-f32-multisteps-bs72-e35 | 64.7513 | 89.4906 | 85,316,861,952 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-f64-multisteps-syncbn-bs36-e35 | 67.9878 | 91.3194 | 170,633,723,904 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-tp-f8-cosine-bs72-e75 | 49.3262 | 78.3795 | 12,125,732,864 | 15,425,559 | 75|
| mini_st2stv2-s3d-resnet-18-tp-f16-multisteps-bs72-e35 | 55.3521 | 83.524 | 24,251,465,728 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-tp-f32-multisteps-bs72-e35 | 60.0305 | 86.7531 | 48,502,931,456 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-tp-f64-multisteps-syncbn-bs36-e35 | 67.2595 | 90.7774 | 97,005,862,912 | 15,425,559 | 35|
| | | | | |
| mini_st2stv2-s3d-resnet-50-f8-cosine-bs72-e75 | 61.3103 | 86.6599 | 39,517,814,784 | 27,485,079 | 75|
| mini_st2stv2-s3d-resnet-50-f16-multisteps-bs48-e35 | 65.5734 | 89.9737 | 79,035,629,568 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 67.2341 | 90.0237 | 158,071,259,136 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 69.148 | 91.8022 | 316,142,518,272 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-tp-f8-cosine-bs72-e75 | 49.555 | 77.7863 | 19,672,203,264 | 27,485,079 | 75|
| mini_st2stv2-s3d-resnet-50-tp-f16-multisteps-bs60-e35 | 56.6658 | 84.3631 | 39,344,406,528 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-tp-f32-multisteps-syncbn-bs36-e35 | 62.4915 | 87.7286 | 78,688,813,056 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e35 | 68.3858 | 91.455 | 157,377,626,112 | 27,485,079 | 35|
| | | | | |
| | | | | |
| mini_st2stv2-i3d-f8-cosine-bs72-e75 | 56.3607 | 82.49 | 33,573,614,592 | 12,376,439 | 75 |
| mini_st2stv2-i3d-f16-multisteps-bs72-e35 | 61.8188 | 87.0836 | 67,380,196,352 | 12,376,439 | 35 |
| mini_st2stv2-i3d-f32-multisteps-bs48-e35 | 63.4969 | 88.2702 | 134,993,359,872 | 12,376,439 | 35 |
| mini_st2stv2-i3d-f64-multisteps-syncbn-bs36-e35 | 67.8862 | 90.9045 | 270,452,654,080 | 12,376,439 | 35|
| mini_st2stv2-i3d-tp-max-f8-cosine-bs72-e75 | 50.8094 | 78.3541 | 21,411,353,600 | 12,376,439 | 75 |
| mini_st2stv2-i3d-tp-max-f16-multisteps-bs72-e35 | 57.1489 | 84.2105 | 42,822,707,200 | 12,376,439 | 35 |
| mini_st2stv2-i3d-tp-max-f32-multisteps-bs72-e35 | 61.5984 | 87.2108 | 85,645,414,400 | 12,376,439 | 35 |
| mini_st2stv2-i3d-tp-max-f64-multisteps-syncbn-bs36-e35 | 67.8269 | 91.3873 | 171,290,828,800 | 12,376,439 | 35|
| | | | | |
| mini_st2stv2-i3d-resnet-18-f8-cosine-bs72-e75 | 55.4454 | 81.7696 | 43,217,190,912 | 33,210,903 | 75 |
| mini_st2stv2-i3d-resnet-18-f16-multisteps-bs72-e35 | 61.412 | 86.6429 | 86,434,381,824 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-f32-multisteps-bs72-e35 | 62.768 | 87.2955 | 172,868,763,648 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-f64-multisteps-bs48-e35 | 65.1411 | 89.1008 | 345,737,527,296 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-tp-max-f8-cosine-bs72-e75 | 47.7837 | 76.8963 | 22,472,425,472 | 33,210,903 | 75 |
| mini_st2stv2-i3d-resnet-18-tp-max-f16-multisteps-bs72-e35 | 53.8266 | 81.0662 | 44,944,850,944 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-tp-max-f32-multisteps-bs72-e35 | 57.2252 | 84.6258 | 89,889,701,888 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-tp-max-f64-multisteps-bs60-e35 | 64.3783 | 88.4651 | 179,779,403,776 | 33,210,903 | 35 |
| | | | | |
| mini_st2stv2-i3d-resnet-50-f8-cosine-bs72-e75 | 62.6324 | 86.27 | 64,180,322,304 | 46,339,607 | 75 |
| mini_st2stv2-i3d-resnet-50-f16-multisteps-bs60-e35 | 66.209 | 89.2872 | 128,360,644,608 | 46,339,607 | 35 |
| mini_st2stv2-i3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 67.5643 | 89.939 | 256,721,289,216 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 68.9702 | 91.4719 | 513,442,578,432 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-tp-f8-cosine-bs60-e75 | 53.4706 | 79.8797 | 29,460,135,936 | 46,339,607 | 75 |
| mini_st2stv2-i3d-resnet-50-tp-f16-multisteps-bs60-e35 | 59.9034 | 85.414 | 58,920,271,872 | 46,339,607 | 35 |
| mini_st2stv2-i3d-resnet-50-tp-f32-multisteps-syncbn-bs36-e35 | 63.8621 | 88.3409 | 117,840,543,744 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e35 | 69.0973 | 91.7599 | 235,681,087,488 | 46,339,607 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-inception-v1-f8-cosine-bs72-e75 | 59.7424 | 85.948 | 12,047,006,720 | 5,706,503 | 75 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e35 | 63.9376 | 89.0245 | 24,094,013,440 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e35 | 65.3279 | 89.1628 | 48,188,026,880 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e35 | 67.5898 | 90.9637 | 96,376,053,760 | 5,706,503 | 35|
| mini_st2stv2-TAM-b3-sum-inception-v1-tp-max-f8-cosine-bs72-e75 | 50.6992 | 78.6423 | 6,570,863,936 | 5,706,503 | 75|
| mini_st2stv2-TAM-b3-sum-inception-v1-tp-max-f16-multisteps-bs72-e35 | 56.1658 | 83.8715 | 13,141,727,872 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-tp-max-f32-multisteps-syncbn-bs36-e35 | 60.2101 | 86.9004 | 26,283,455,744 | 5,706,503 | 35|
| mini_st2stv2-TAM-b3-sum-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e35 | 66.3364 | 90.4726 | 52,566,911,488 | 5,706,503 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-resnet-18-f8-cosine-bs72-e75 | 59.1406 | 85.092 | 14,530,768,896 | 11,225,559 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e35 | 62.0561 | 87.6684 | 29,061,537,792 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e35 | 63.1409 | 87.8464 | 58,123,075,584 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e35 | 64.6416 | 89.2984 | 116,246,151,168 | 11,225,559 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-18-tp-max-f8-cosine-bs72-e75 | 48.0973 | 77.3625 | 7,535,155,712 | 11,225,559 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-18-tp-max-f16-multisteps-bs72-e35 | 53.174 | 81.3204 | 15,070,311,424 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-tp-max-f32-multisteps-bs72-e35 | 56.5811 | 84.4902 | 30,140,622,848 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-tp-max-f64-multisteps-syncbn-bs36-e35 | 63.413 | 88.9764 | 60,281,245,696 | 11,225,559 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-resnet-50-f8-cosine-bs72-e75 | 65.4378 | 89.1432 | 32,831,963,136 | 23,725,655 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-50-f16-multisteps-bs48-e35 | 68.6329 | 90.8467 | 65,663,926,272 | 23,725,655 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e35 | 69.3089 | 91.4464 | 131,327,852,544 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e35 | 71.5532 | 92.5052 | 262,655,705,088 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-tp-max-f8-cosine-bs72-e75 | 54.5216 | 81.4476 | 14,213,054,464 | 23,725,655 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-50-tp-max-f16-multisteps-bs60-e35 | 59.1067 | 85.7615 | 28,426,108,928 | 23,725,655 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e35 | 63.042 | 88.4231 | 56,852,217,856 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e35 | 69.1819 | 91.9292 | 113,704,435,712 | 23,725,655 | 35|

### Dataset:  Mini-Kinetics400

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs |
|------------|-------|-------|-------|--------|--------|
| mini_kinetics400-inception-v1-f8-cosine-bs72-e100 | 70.4178 | 88.9296 | 11,978,817,536 | 5,804,904 | 100|
| mini_kinetics400-inception-v1-f16-multisteps-bs72-e45 | 70.5195 | 88.8991 | 23,957,635,072 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-f32-multisteps-bs48-e45 | 70.4076 | 89.0007 | 47,915,270,144 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-f64-multisteps-syncbn-bs36-e45 | 70.2439 | 89.0447 | 95,830,540,288 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-tp-max-f8-cosine-bs72-e100 | 64.6742 | 85.6664 | 6,521,725,952 | 5,804,904 | 100|
| mini_kinetics400-inception-v1-tp-max-f16-multisteps-bs72-e45 | 67.0326 | 87.1302 | 13,043,451,904 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-tp-max-f32-multisteps-bs72-e45 | 71.0786 | 89.6107 | 26,086,903,808 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e45 | 73.3943 | 90.691 | 52,173,807,616 | 5,804,904 | 45|
| | | | | |
| mini_kinetics400-resnet-18-f8-cosine-bs72-e100 | 67.8662 | 87.3437 | 14,508,490,752 | 11,279,112 | 100|
| mini_kinetics400-resnet-18-f16-multisteps-bs72-e45 | 68.4965 | 87.425 | 29,016,981,504 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-f32-multisteps-bs72-e45 | 69.0759 | 88.0655 | 58,033,963,008 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-f64-multisteps-syncbn-bs36-e45 | 68.6281 | 87.7338 | 116,067,926,016 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-tp-f8-cosine-bs72-e100 | 61.6041 | 82.9115 | 7,520,780,288 | 11,279,112 | 100|
| mini_kinetics400-resnet-18-tp-f16-multisteps-bs72-e45 | 64.1862 | 85.4122 | 15,041,560,576 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-tp-f32-multisteps-bs72-e45 | 68.4863 | 88.2891 | 30,083,121,152 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-tp-f64-multisteps-bs48-e45 | 71.0887 | 89.7835 | 60,166,242,304 | 11,279,112 | 45|
| | | | | |
| mini_kinetics400-resnet-50-f8-cosine-bs72-e100 | 72.1358 | 90.1698 | 32,697,090,048 | 23,917,832 | 100|
| mini_kinetics400-resnet-50-f16-multisteps-bs48-e45 | 72.4713 | 89.875 | 65,394,180,096 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-f32-multisteps-syncbn-bs36-e45 | 73.4654 | 90.3963 | 130,788,360,192 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-f64-multisteps-syncbn-bs36-e45 | 73.4511 | 90.3311 | 261,576,720,384 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-tp-max-f8-cosine-bs72-e100 | 66.9411 | 86.5304 | 14,135,984,128 | 23,917,832 | 100|
| mini_kinetics400-resnet-50-tp-max-f16-multisteps-bs72-e45 | 70.1332 | 88.7567 | 28,271,968,256 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e45 | 72.9167 | 90.4573 | 56,543,936,512 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e45 | 75.2489 | 91.8648 | 113,087,873,024 | 23,917,832 | 45|
| | | | | |
| | | | | |
| mini_kinetics400-s3d-f8-cosine-bs72-e100 | 67.0428 | 86.7846 | 19,778,626,560 | 8,108,328 | 100|
| mini_kinetics400-s3d-f16-multisteps-bs72-e45 | 69.7062 | 88.8177 | 39,712,547,840 | 8,108,328 | 45|
| mini_kinetics400-s3d-f32-multisteps-syncbn-bs36-e45 | 72.5 | 90.5793 | 79,580,390,400 | 8,108,328 | 45|
| mini_kinetics400-s3d-f64-multisteps-syncbn-bs36-e45 | 74.7309 | 91.5804 | 159,316,075,520 | 8,108,328 | 45|
| mini_kinetics400-s3d-tp-f8-cosine-bs72-e100 | 61.4822 | 83.5722 | 12,264,670,208 | 8,108,328 | 100|
| mini_kinetics400-s3d-tp-f16-multisteps-bs72-e45 | 65.8331 | 86.3576 | 24,529,340,416 | 8,108,328 | 45|
| mini_kinetics400-s3d-tp-f32-multisteps-bs60-e45 | 69.7672 | 88.8686 | 49,058,680,832 | 8,108,328 | 45|
| mini_kinetics400-s3d-tp-max-f64-multisteps-syncbn-bs36-e45 | 73.2012 | 90.8638 | 98,117,361,664 | 8,108,328 | 45|
| | | | | |
| mini_kinetics400-s3d-resnet-18-f8-cosine-bs72-e100 | 67.8357 | 87.4657 | 21,329,215,488 | 15,483,528 | 100|
| mini_kinetics400-s3d-resnet-18-f16-multisteps-bs72-e45 | 70.3365 | 89.5090 | 42,658,430,976 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-f32-multisteps-bs72-e45 | 72.8881 | 90.7899 | 85,316,861,952 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-f64-multisteps-syncbn-bs36-e45 | 74.4512 | 91.6464 | 170,633,723,904 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-tp-f8-cosine-bs72-e100 | 59.571 | 82.3828 | 12,125,732,864 | 15,483,528 | 100|
| mini_kinetics400-s3d-resnet-18-tp-f16-multisteps-bs72-e45 | 63.3222 | 85.1581 | 24,251,465,728 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-tp-f32-multisteps-bs72-e45 | 68.0797 | 87.7097 | 48,502,931,456 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-tp-f64-multisteps-bs48-e45 | 71.1294 | 90.0579 | 97,005,862,912 | 15,483,528 | 45|
| | | | | |
| mini_kinetics400-s3d-resnet-50-f8-cosine-bs60-e100 | 71.8715 | 89.814 | 39,517,814,784 | 27,716,616 | 100|
| mini_kinetics400-s3d-resnet-50-f16-multisteps-bs48-e45 | 73.6607 | 91.0135 | 79,035,629,568 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-f32-multisteps-syncbn-bs36-e45 | 76.3516 | 92.2764 | 158,071,259,136 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-f64-multisteps-syncbn-bs36-e45 | 77.1582 | 92.8703 | 316,142,518,272 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-tp-f8-cosine-bs72-e100 | 64.4607 | 85.1784 | 19,672,203,264 | 27,716,616 | 100|
| mini_kinetics400-s3d-resnet-50-tp-f16-multisteps-bs72-e45 | 69.0759 | 88.2383 | 39,344,406,528 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-tp-f32-multisteps-syncbn-bs36-e45 | 72.2866 | 90.4167 | 78,688,813,056 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e45* | 74.9898 | 91.9207 | 157,377,626,112 | 27,716,616 | 43|
| | | | | |
| | | | | |
| mini_kinetics400-i3d-f8-cosine-bs72-e100 | 68.0899 | 87.6182 | 33,573,614,592 | 12,492,264 | 100|
| mini_kinetics400-i3d-f16-multisteps-bs72-e45 | 70.8651 | 89.387 | 67,380,196,352 | 12,492,264 | 45|
| mini_kinetics400-i3d-f32-multisteps-syncbn-bs36-e45 | 73.6687 | 91.1382 | 135,226,327,040 | 12,492,264 | 45|
| mini_kinetics400-i3d-f64-multisteps-syncbn-bs36-e45 | 74.9543 | 91.6921 | 270,452,654,080 | 12,492,264 | 45|
| mini_kinetics400-i3d-tp-max-f8-cosine-bs72-e100 | 62.3869 | 84.0398 | 21,411,353,600 | 12,492,264 | 100|
| mini_kinetics400-i3d-tp-max-f16-multisteps-bs72-e45 | 66.3414 | 86.9472 | 42,822,707,200 | 12,492,264 | 45|
| mini_kinetics400-i3d-tp-max-f32-multisteps-bs72-e45 | 71.1192 | 89.0312 | 85,645,414,400 | 12,492,264 | 45|
| mini_kinetics400-i3d-tp-max-f64-multisteps-syncbn-bs36-e45 | 73.1301 | 91.1281 | 171,290,828,800 | 12,492,264 | 45|
| | | | | |
| mini_kinetics400-i3d-resnet-18-f8-cosine-bs72-e100 | 66.5345 | 86.7948 | 43,217,190,912 | 33,268,872 | 100|
| mini_kinetics400-i3d-resnet-18-f16-multisteps-bs72-e45 | 70.367 | 89.1939 | 86,434,381,824 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-f32-multisteps-bs72-e45 | 72.7559 | 90.3019 | 172,868,763,648 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-f64-multisteps-bs60-e45 | 73.437 | 90.6476 | 345,737,527,296 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-tp-max-f8-cosine-bs72-e100 | 58.6154 | 81.5493 | 22,472,425,472 | 33,268,872 | 100|
| mini_kinetics400-i3d-resnet-18-tp-max-f16-multisteps-bs72-e45 | 63.068 | 84.7616 | 44,944,850,944 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-tp-max-f32-multisteps-bs72-e45 | 66.6972 | 86.9066 | 89,889,701,888 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-tp-max-f64-multisteps-bs60-e45 | 71.0074 | 89.0922 | 179,779,403,776 | 33,268,872 | 45|
| | | | | |
| mini_kinetics400-i3d-resnet-50-f8-cosine-bs72-e100 | 73.3455 | 90.4442 | 64,180,322,304 | 46,571,144 | 100|
| mini_kinetics400-i3d-resnet-50-f16-multisteps-bs60-e45 | 75.5718 | 91.8573 | 128,360,644,608 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-f32-multisteps-syncbn-bs36-e45 | 77.1646 | 92.6016 | 256,721,289,216 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-f64-multisteps-syncbn-bs36-e45* | 77.6965 | 92.8601 | 513,442,578,432 | 46,571,144 | 41|
| mini_kinetics400-i3d-resnet-50-tp-max-f8-cosine-bs60-e100 | 65.0605 | 85.6155 | 29,460,135,936 | 46,571,144 | 100|
| mini_kinetics400-i3d-resnet-50-tp-max-f16-multisteps-bs60-e45 | 70.0722 | 88.5941 | 58,920,271,872 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e45 | 73.2012 | 90.5996 | 117,840,543,744 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e45 | 75.9191 | 92.0172 | 235,681,087,488 | 46,571,144 | 45|
| | | | | |
| | | | | |
| mini_kinetics400-TAM-b3-sum-inception-v1-f8-cosine-bs72-e100 | 68.7805 | 88.2114 | 12,047,006,720 | 5,822,328 | 100|
| mini_kinetics400-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e45 | 70.0407 | 89.1565 | 24,094,013,440 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e45 | 72.9878 | 91.1585 | 48,188,026,880 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e45 | 74.1057 | 91.5752 | 96,376,053,760 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-tp-max-f8-cosine-bs72-e100 | 58.313 | 81.626 | 6,570,863,936 | 5,822,328 | 100|
| mini_kinetics400-TAM-b3-sum-inception-v1-tp-max-f16-multisteps-bs72-e45 | 61.4431 | 84.4106 | 13,141,727,872 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-tp-max-f32-multisteps-syncbn-bs36-e45 | 66.7276 | 87.5915 | 26,283,455,744 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e45 | 70.3557 | 89.6545 | 52,566,911,488 | 5,822,328 | 45|
| | | | | |
| mini_kinetics400-TAM-b3-sum-resnet-18-f8-cosine-bs72-e100 | 69.0759 | 88.3501 | 14,530,768,896 | 11,283,528 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e45 | 71.25 | 89.502 | 29,061,537,792 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e45 | 72.6118 | 90.3455 | 58,123,075,584 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e45 | 72.937 | 90.7724 | 116,246,151,168 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-tp-max-f8-cosine-bs72-e100 | 59.1847 | 82.2914 | 7,535,155,712 | 11,283,528 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-18-tp-max-f16-multisteps-bs72-e45 | 62.6423 | 84.624 | 15,070,311,424 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-tp-max-f32-multisteps-bs72-e45 | 67.2053 | 87.439 | 30,140,622,848 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-tp-max-f64-multisteps-syncbn-bs36-e45 | 69.9797 | 89.187 | 60,281,245,696 | 11,283,528 | 45|
| | | | | |
| mini_kinetics400-TAM-b3-sum-resnet-50-f8-cosine-bs72-e100 | 74.065 | 90.3557 | 32,831,963,136 | 23,957,192 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-50-f16-multisteps-bs48-e45 | 76.4329 | 92.0427 | 65,663,926,272 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e45 | 77.8353 | 92.8963 | 131,327,852,544 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e45 | 78.3252 | 93.4105 | 262,655,705,088 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-tp-max-f8-cosine-bs72-e100 | 67.3374 | 87.124 | 14,213,054,464 | 23,957,192 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-50-tp-max-f16-multisteps-bs60-e45 | 70.3963 | 88.9634 | 28,426,108,928 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e45 | 73.3334 | 90.8943 | 56,852,217,856 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e45 | 75.8684 | 92.1694 | 113,704,435,712 | 23,957,192 | 45|




### Dataset:  Mini-Moments

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs |
|------------|-------|-------|-------|--------|--------|
| mini_moments-inception-v1-f8-cosine-bs72-e75 | 21.73 | 44.46 | 11,978,817,536 | 5,804,904 | 75|
| mini_moments-inception-v1-f16-multisteps-bs72-e35 | 23.3 | 46.52 | 23,957,635,072 | 5,804,904 | 35|
| mini_moments-inception-v1-f32-multisteps-bs48-e35 | 23.09 | 46.54 | 47,915,270,144 | 5,804,904 | 35|
| mini_moments-inception-v1-f64-multisteps-syncbn-bs36-e35 | 23.0815 | 46.6427 | 95,830,540,288 | 5,804,904 | 35|
| mini_moments-inception-v1-tp-f8-cosine-bs72-e75 | 22.85 | 46.01 | 6,521,725,952 | 5,804,904 | 75|
| mini_moments-inception-v1-tp-f16-multisteps-bs72-e35 | 24.59 | 49.7 | 13,043,451,904 | 5,804,904 | 35|
| mini_moments-inception-v1-tp-f32-multisteps-bs72-e35 | 25.63 | 51.24 | 26,086,903,808 | 5,804,904 | 35|
| mini_moments-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e35 | 26.1291 | 51.7386 | 52,173,807,616 | 5,804,904 | 35|
| | | | | |
| mini_moments-resnet-18-f8-cosine-bs72-e75 | 20.55 | 40.69 | 14,508,490,752 | 11,279,112 | 75|
| mini_moments-resnet-18-f16-multisteps-bs72-e35 | 21.23 | 42.38 | 29,016,981,504 | 11,279,112 | 35|
| mini_moments-resnet-18-f32-multisteps-bs72-e35 | 20.92 | 41.65 | 58,033,963,008 | 11,279,112 | 35|
| mini_moments-resnet-18-f64-multisteps-syncbn-bs36-e35 | 20.8958 | 42.7514 | 116,067,926,016 | 11,279,112 | 35|
| mini_moments-resnet-18-tp-max-f8-cosine-bs72-e75 | 20.95 | 43.03 | 7,520,780,288 | 11,279,112 | 75|
| mini_moments-resnet-18-tp-max-f16-multisteps-bs72-e35 | 23.48 | 47.44 | 15,041,560,576 | 11,279,112 | 35|
| mini_moments-resnet-18-tp-max-f32-multisteps-bs72-e35 | 23.74 | 47.67 | 30,083,121,152 | 11,279,112 | 35|
| mini_moments-resnet-18-tp-max-f64-multisteps-bs48-e35 | 24.79 | 49.11 | 60,166,242,304 | 11,279,112 | 35|
| | | | | |
| mini_moments-resnet-50-f8-cosine-bs72-e75 | 22.57 | 44.46 | 32,697,090,048 | 23,917,832 | 75|
| mini_moments-resnet-50-f16-multisteps-bs60-e35 | 23.65 | 47.15 | 65,394,180,096 | 23,917,832 | 35|
| mini_moments-resnet-50-f32-multisteps-syncbn-bs36-e35 | 23.8909 | 47.542 | 130,788,360,192 | 23,917,832 | 35|
| mini_moments-resnet-50-f64-multisteps-syncbn-bs36-e35 | 23.8909 | 47.6319 | 261,576,720,384 | 23,917,832 | 35|
| mini_moments-resnet-50-tp-max-f8-cosine-bs72-e75 | 23.48 | 46.33 | 14,135,984,128 | 23,917,832 | 75|
| mini_moments-resnet-50-tp-max-f16-multisteps-bs72-e35 | 25.53 | 50.64 | 28,271,968,256 | 23,917,832 | 35|
| mini_moments-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e35 | 27.1783 | 52.6879 | 56,543,936,512 | 23,917,832 | 35|
| mini_moments-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e35 | 27.7978 | 54.0867 | 113,087,873,024 | 23,917,832 | 35|
| | | | | |
| | | | | |
| mini_moments-s3d-f8-cosine-bs72-e75 | 21.56 | 44.42 | 19,778,626,560 | 8,108,328 | 75|
| mini_moments-s3d-f16-multisteps-bs72-e35 | 24.45 | 49.27 | 39,712,547,840 | 8,108,328 | 35|
| mini_moments-s3d-f32-multisteps-syncbn-bs36-e35 | 25.4097 | 51.0492 | 79,580,390,400 | 8,108,328 | 35|
| mini_moments-s3d-f64-multisteps-syncbn-bs36-e35 | 26.5987 | 52.3082 | 159,316,075,520 | 8,108,328 | 35|
| mini_moments-s3d-tp-f8-cosine-bs72-e75 | 20.53 | 42.37 | 12,264,670,208 | 8,108,328 | 75|
| mini_moments-s3d-tp-f16-multisteps-bs72-e35 | 24.07 | 48.02 | 24,529,340,416 | 8,108,328 | 35|
| mini_moments-s3d-tp-f32-multisteps-bs60-e35 | 25.79 | 51.08 | 49,058,680,832 | 8,108,328 | 35|
| mini_moments-s3d-tp-max-f64-multisteps-syncbn-bs36-e35 | 26.3789 | 52.2782 | 98,117,361,664 | 8,108,328 | 35|
| | | | | |
| mini_moments-s3d-resnet-18-f8-cosine-bs72-e75 | 22.06 | 45.2 | 21,329,215,488 | 15,483,528 | 75|
| mini_moments-s3d-resnet-18-f16-multisteps-bs72-e35 | 24.55 | 49.22 | 42,658,430,976 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-f32-multisteps-bs72-e35 | 25.6 | 50.74 | 85,316,861,952 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-f64-multisteps-syncbn-bs36-e35 | 25.6695 | 51.1791 | 170,633,723,904 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-tp-f8-cosine-bs72-e75 | 19.12 | 40.73 | 12,125,732,864 | 15,483,528 | 75|
| mini_moments-s3d-resnet-18-tp-f16-multisteps-bs72-e35 | 22.14 | 45.71 | 24,251,465,728 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-tp-f32-multisteps-bs72-e35 | 23.73 | 48.41 | 48,502,931,456 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-tp-f64-multisteps-bs48-e35 | 24.95 | 49.74 | 97,005,862,912 | 15,483,528 | 35|
| | | | | |
| mini_moments-s3d-resnet-50-f8-cosine-bs72-e75 | 24.42 | 48.6 | 39,517,814,784 | 27,716,616 | 75|
| mini_moments-s3d-resnet-50-f16-multisteps-bs48-e35 | 26.99 | 52.72 | 79,035,629,568 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 27.9576 | 53.5971 | 158,071,259,136 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 27.9477 | 54.2965 | 316,142,518,272 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-tp-f8-cosine-bs72-e75 | 21.08 | 44.15 | 19,672,203,264 | 27,716,616 | 75|
| mini_moments-s3d-resnet-50-tp-f16-multisteps-bs60-e35 | 24.55 | 50.1 | 39,344,406,528 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-tp-f32-multisteps-syncbn-bs36-e35 | 26.6287 | 52.2982 | 78,688,813,056 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e35 | 27.7578 | 53.9568 | 157,377,626,112 | 27,716,616 | 35|
| | | | | |
| | | | | |
| mini_moments-i3d-f8-cosine-bs60-e75 | 22.42 | 45.54 | 33,573,614,592 | 12,492,264 | 75|
| mini_moments-i3d-f16-multisteps-bs60-e35 | 25.56 | 50.27 | 67,380,196,352 | 12,492,264 | 35|
| mini_moments-i3d-f32-multisteps-bs48-e35 | 26.17 | 51.82 | 135,226,327,040 | 12,492,264 | 35|
| mini_moments-i3d-f64-multisteps-syncbn-bs36-e35 | 26.4388 | 52.2782 | 270,452,654,080 | 12,492,264 | 35|
| mini_moments-i3d-tp-max-f8-cosine-bs72-e75 | 21.83 | 44.08 | 21,411,353,600 | 12,492,264 | 75|
| mini_moments-i3d-tp-max-f16-multisteps-bs72-e35 | 24.66 | 48.75 | 42,822,707,200 | 12,492,264 | 35|
| mini_moments-i3d-tp-max-f32-multisteps-bs72-e35 | 25.94 | 51.47 | 85,645,414,400 | 12,492,264 | 35|
| mini_moments-i3d-tp-max-f64-multisteps-syncbn-bs36-e35 | 26.6687 | 52.5979 | 171,290,828,800 | 12,492,264 | 35|
| | | | | |
| mini_moments-i3d-resnet-18-f8-cosine-bs72-e75 | 20.9 | 41.76 | 43,217,190,912 | 33,268,872 | 75|
| mini_moments-i3d-resnet-18-f16-multisteps-bs72-e35 | 22.31 | 45.11 | 86,434,381,824 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-f32-multisteps-bs72-e35 | 22.66 | 45.71 | 172,868,763,648 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-f64-multisteps-bs60-e35 | 22.7 | 45.94 | 345,737,527,296 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-tp-max-f8-cosine-bs72-e75 | 19.35 | 39.83 | 22,472,425,472 | 33,268,872 | 75|
| mini_moments-i3d-resnet-18-tp-max-f16-multisteps-bs72-e35 | 21.66 | 43.81 | 44,944,850,944 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-tp-max-f32-multisteps-bs72-e35 | 22.32 | 45.8 | 89,889,701,888 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-tp-max-f64-multisteps-bs60-e35 | 23.36 | 46.01 | 179,779,403,776 | 33,268,872 | 35|
| | | | | |
| mini_moments-i3d-resnet-50-f8-cosine-bs72-e75 | 24.57 | 48.1 | 64,180,322,304 | 46,571,144 | 75|
| mini_moments-i3d-resnet-50-f16-multisteps-bs60-e35 | 26.51 | 52.03 | 128,360,644,608 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 27.498 | 53.4672 | 256,721,289,216 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 27.6279 | 53.2774 | 513,442,578,432 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-tp-max-f8-cosine-bs60-e75 | 22.64 | 44.34 | 29,460,135,936 | 46,571,144 | 75|
| mini_moments-i3d-resnet-50-tp-max-f16-multisteps-bs60-e35 | 25.41 | 50.51 | 58,920,271,872 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-tp-f32-multisteps-syncbn-bs36-e35 | 27.1346 | 52.9494 | 117,840,543,744 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e35 | 27.8377 | 53.9768 | 235,681,087,488 | 46,571,144 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-inception-v1-f8-cosine-bs72-e75 | 23.2953 | 47.2905 | 12,047,006,720 | 5,822,328 | 75 |
| mini_moments-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e35 | 25.9148 | 50.9398 | 24,094,013,440 | 5,822,328 | 35 |
| mini_moments-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e35 | 26.3247 | 52.1395 | 48,188,026,880 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e35 | 26.7986 | 53.0975 | 96,376,053,760 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-tp-max-f8-cosine-bs72-e75 | 21.4757 | 44.3111 | 6,570,863,936 | 5,822,328 | 75 |
| mini_moments-TAM-b3-sum-inception-v1-tp-max-f16-multisteps-bs72-e35 | 23.8852 | 48.1404 | 13,141,727,872 | 5,822,328 | 35 |
| mini_moments-TAM-b3-sum-inception-v1-tp-max-f32-multisteps-syncbn-bs36-e35 | 25.6449 | 50.9798 | 26,283,455,744 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-tp-max-f64-multisteps-syncbn-bs36-e35 | 26.5687 | 52.6379 | 52,566,911,488 | 5,822,328 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-resnet-18-f8-cosine-bs72-e75 | 22.0556 | 44.881 | 14,530,768,896 | 11,283,528 | 75 |
| mini_moments-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e35 | 24.1252 | 48.3003 | 29,061,537,792 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e35 | 24.4451 | 48.8702 | 58,123,075,584 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e35 | 24.3751 | 49.74 | 116,246,151,168 | 11,283,528 | 35|
| mini_moments-TAM-b3-sum-resnet-18-tp-max-f8-cosine-bs72-e75 | 21.58 | 44.92 | 7,535,155,712 | 11,283,528 | 75 |
| mini_moments-TAM-b3-sum-resnet-18-tp-max-f16-multisteps-bs72-e35 | 23.79 | 48.02 | 15,070,311,424 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-tp-max-f32-multisteps-bs72-e35 | 24.775 | 49.4001 | 30,140,622,848 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-tp-max-f64-multisteps-syncbn-bs36-e35 | 25.4949 | 51.3497 | 60,281,245,696 | 11,283,528 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-resnet-50-f8-cosine-bs72-e75 | 25.9748 | 50.08 | 32,831,963,136 | 23,957,192 | 75 |
| mini_moments-TAM-b3-sum-resnet-50-f16-multisteps-bs36-e35 | 28.2344 | 54.809 | 65,663,926,272 | 23,957,192 | 35 |
| mini_moments-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e35 | 28.8269 | 55.6055 | 131,327,852,544 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e35 | 28.8569 | 55.2658 | 262,655,705,088 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-tp-max-f8-cosine-bs72-e75 | 24.8 | 48.85 | 14,213,054,464 | 23,957,192 | 75 |
| mini_moments-TAM-b3-sum-resnet-50-tp-max-f16-multisteps-bs60-e35 | 26.9446 | 52.4495 | 28,426,108,928 | 23,957,192 | 35 |
| mini_moments-TAM-b3-sum-resnet-50-tp-max-f32-multisteps-syncbn-bs36-e35 | 28.6371 | 55.1259 | 56,852,217,856 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-tp-max-f64-multisteps-syncbn-bs36-e35 | 29.2766 | 55.3957 | 113,704,435,712 | 23,957,192 | 35|

The model name should self-explain the experimental setting

### Dataset:  Mini-Something-Something-V2

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs | 
|------------|-------|-------|-------|--------|--------|
| mini_st2stv2-inception-v1-f8-cosine-bs72-e75 | 33.1384 | 63.6156 | 11,978,817,536 | 5,689,079 | 75|
| mini_st2stv2-inception-v1-f16-multisteps-bs72-e35 | 34.6555 | 65.2852 | 23,957,635,072 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-f32-multisteps-bs48-e35 | 34.5199 | 66.3022 | 47,915,270,144 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-f64-multisteps-syncbn-bs36-e35 | 36.3228 | 67.4119 | 95,830,540,288 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-ts-f8-cosine-bs72-e75 | 53.2672 | 81.0153 | 6,521,725,952 | 5,689,079 | 75|
| mini_st2stv2-inception-v1-ts-f16-multisteps-bs72-e35 | 54.5555 | 82.9138 | 13,043,451,904 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-ts-f32-multisteps-bs72-e35 | 55.1064 | 83.7105 | 26,086,903,808 | 5,689,079 | 35|
| mini_st2stv2-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e35 | 57.4187 | 85.6877 | 52,173,807,616 | 5,689,079 | 35|
| | | | | |
| mini_st2stv2-resnet-18-f8-cosine-bs72-e75 | 29.5534 | 59.4118 | 14,508,490,752 | 11,221,143 | 75|
| mini_st2stv2-resnet-18-f16-multisteps-bs72-e35 | 30.9263 | 61.5815 | 29,016,981,504 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-f32-multisteps-bs72-e35 | 30.9348 | 61.2255 | 58,033,963,008 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-f64-multisteps-syncbn-bs36-e35 | 32.08 | 62.9573 | 116,067,926,016 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-ts-f8-cosine-bs72-e75 | 50.0636 | 78.7016 | 7,520,780,288 | 11,221,143 | 75|
| mini_st2stv2-resnet-18-ts-f16-multisteps-bs72-e35 | 51.6145 | 81.473 | 15,041,560,576 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-ts-f32-multisteps-bs72-e35 | 52.4451 | 81.9561 | 30,083,121,152 | 11,221,143 | 35|
| mini_st2stv2-resnet-18-ts-f64-multisteps-bs48-e35 | 54.7928 | 83.8546 | 60,166,242,304 | 11,221,143 | 35|
| | | | | |
| mini_st2stv2-resnet-50-f8-cosine-bs72-e75 | 33.9351 | 63.9885 | 32,697,090,048 | 23,686,295 | 75|
| mini_st2stv2-resnet-50-f16-multisteps-bs60-e35 | 35.342 | 66.8785 | 65,394,180,096 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-f32-multisteps-syncbn-bs36-e35 | 36.2551 | 67.6575 | 130,788,360,192 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-f64-multisteps-syncbn-bs36-e35 | 36.7971 | 68.1233 | 261,576,720,384 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-ts-f8-cosine-bs72-e75 | 53.4961 | 80.2526 | 14,135,984,128 | 23,686,295 | 75|
| mini_st2stv2-resnet-50-ts-f16-multisteps-bs72-e35 | 55.8946 | 83.4901 | 28,271,968,256 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e35 | 56.9225 | 84.5873 | 56,543,936,512 | 23,686,295 | 35|
| mini_st2stv2-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e35 | 59.3157 | 86.8903 | 113,087,873,024 | 23,686,295 | 35|
| | | | | |
| | | | | |
| mini_st2stv2-s3d-f8-cosine-bs72-e75 | 57.115 | 84.2105 | 19,778,626,560 | 7,992,503 | 75|
| mini_st2stv2-s3d-f16-multisteps-bs72-e35 | 62.7426 | 87.7617 | 39,712,547,840 | 7,992,503 | 35|
| mini_st2stv2-s3d-f32-multisteps-syncbn-bs36-e35 | 65.0576 | 89.2446 | 79,580,390,400 | 7,992,503 | 35|
| mini_st2stv2-s3d-f64-multisteps-syncbn-bs36-e35 | 67.8438 | 91.1331 | 159,316,075,520 | 7,992,503 | 35|
| mini_st2stv2-s3d-ts-f8-cosine-bs72-e75 | 50.8941 | 78.8287 | 12,264,670,208 | 7,992,503 | 75|
| mini_st2stv2-s3d-ts-f16-multisteps-bs72-e35 | 57.115 | 84.5156 | 24,529,340,416 | 7,992,503 | 35|
| mini_st2stv2-s3d-ts-max-f32-multisteps-bs60-e35| 61.590 | 87.8380 | 49,058,680,832 | 7,992,503 | 35|
| mini_st2stv2-s3d-ts-max-f64-multisteps-syncbn-bs36-e35 | 67.2172 | 91.0993 | 98,117,361,664 | 7,992,503 | 35|
| | | | | |
| mini_st2stv2-s3d-resnet-18-f8-cosine-bs72-e75 | 57.4286 | 83.8715 | 21,329,215,488 | 15,425,559 | 75|
| mini_st2stv2-s3d-resnet-18-f16-multisteps-bs72-e35 | 63.5054 | 88.6007 | 42,658,430,976 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-f32-multisteps-bs72-e35 | 64.7513 | 89.4906 | 85,316,861,952 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-f64-multisteps-syncbn-bs36-e35 | 67.9878 | 91.3194 | 170,633,723,904 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-ts-f8-cosine-bs72-e75 | 49.3262 | 78.3795 | 12,125,732,864 | 15,425,559 | 75|
| mini_st2stv2-s3d-resnet-18-ts-f16-multisteps-bs72-e35 | 55.3521 | 83.524 | 24,251,465,728 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-ts-f32-multisteps-bs72-e35 | 60.0305 | 86.7531 | 48,502,931,456 | 15,425,559 | 35|
| mini_st2stv2-s3d-resnet-18-ts-f64-multisteps-syncbn-bs36-e35 | 67.2595 | 90.7774 | 97,005,862,912 | 15,425,559 | 35|
| | | | | |
| mini_st2stv2-s3d-resnet-50-f8-cosine-bs72-e75 | 61.3103 | 86.6599 | 39,517,814,784 | 27,485,079 | 75|
| mini_st2stv2-s3d-resnet-50-f16-multisteps-bs48-e35 | 65.5734 | 89.9737 | 79,035,629,568 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 67.2341 | 90.0237 | 158,071,259,136 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 69.148 | 91.8022 | 316,142,518,272 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-ts-f8-cosine-bs72-e75 | 49.555 | 77.7863 | 19,672,203,264 | 27,485,079 | 75|
| mini_st2stv2-s3d-resnet-50-ts-f16-multisteps-bs60-e35 | 56.6658 | 84.3631 | 39,344,406,528 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-ts-f32-multisteps-syncbn-bs36-e35 | 62.4915 | 87.7286 | 78,688,813,056 | 27,485,079 | 35|
| mini_st2stv2-s3d-resnet-50-ts-f64-multisteps-syncbn-bs36-e35 | 68.3858 | 91.455 | 157,377,626,112 | 27,485,079 | 35|
| | | | | |
| | | | | |
| mini_st2stv2-i3d-f8-cosine-bs72-e75 | 56.3607 | 82.49 | 33,573,614,592 | 12,376,439 | 75 |
| mini_st2stv2-i3d-f16-multisteps-bs72-e35 | 61.8188 | 87.0836 | 67,380,196,352 | 12,376,439 | 35 |
| mini_st2stv2-i3d-f32-multisteps-bs48-e35 | 63.4969 | 88.2702 | 134,993,359,872 | 12,376,439 | 35 |
| mini_st2stv2-i3d-f64-multisteps-syncbn-bs36-e35 | 67.8862 | 90.9045 | 270,452,654,080 | 12,376,439 | 35|
| mini_st2stv2-i3d-ts-max-f8-cosine-bs72-e75 | 50.8094 | 78.3541 | 21,411,353,600 | 12,376,439 | 75 |
| mini_st2stv2-i3d-ts-max-f16-multisteps-bs72-e35 | 57.1489 | 84.2105 | 42,822,707,200 | 12,376,439 | 35 |
| mini_st2stv2-i3d-ts-max-f32-multisteps-bs72-e35 | 61.5984 | 87.2108 | 85,645,414,400 | 12,376,439 | 35 |
| mini_st2stv2-i3d-ts-max-f64-multisteps-syncbn-bs36-e35 | 67.8269 | 91.3873 | 171,290,828,800 | 12,376,439 | 35|
| | | | | |
| mini_st2stv2-i3d-resnet-18-f8-cosine-bs72-e75 | 55.4454 | 81.7696 | 43,217,190,912 | 33,210,903 | 75 |
| mini_st2stv2-i3d-resnet-18-f16-multisteps-bs72-e35 | 61.412 | 86.6429 | 86,434,381,824 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-f32-multisteps-bs72-e35 | 62.768 | 87.2955 | 172,868,763,648 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-f64-multisteps-bs48-e35 | 65.1411 | 89.1008 | 345,737,527,296 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-ts-max-f8-cosine-bs72-e75 | 47.7837 | 76.8963 | 22,472,425,472 | 33,210,903 | 75 |
| mini_st2stv2-i3d-resnet-18-ts-max-f16-multisteps-bs72-e35 | 53.8266 | 81.0662 | 44,944,850,944 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-ts-max-f32-multisteps-bs72-e35 | 57.2252 | 84.6258 | 89,889,701,888 | 33,210,903 | 35 |
| mini_st2stv2-i3d-resnet-18-ts-max-f64-multisteps-bs60-e35 | 64.3783 | 88.4651 | 179,779,403,776 | 33,210,903 | 35 |
| | | | | |
| mini_st2stv2-i3d-resnet-50-f8-cosine-bs72-e75 | 62.6324 | 86.27 | 64,180,322,304 | 46,339,607 | 75 |
| mini_st2stv2-i3d-resnet-50-f16-multisteps-bs60-e35 | 66.209 | 89.2872 | 128,360,644,608 | 46,339,607 | 35 |
| mini_st2stv2-i3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 67.5643 | 89.939 | 256,721,289,216 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 68.9702 | 91.4719 | 513,442,578,432 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-ts-f8-cosine-bs60-e75 | 53.4706 | 79.8797 | 29,460,135,936 | 46,339,607 | 75 |
| mini_st2stv2-i3d-resnet-50-ts-f16-multisteps-bs60-e35 | 59.9034 | 85.414 | 58,920,271,872 | 46,339,607 | 35 |
| mini_st2stv2-i3d-resnet-50-ts-f32-multisteps-syncbn-bs36-e35 | 63.8621 | 88.3409 | 117,840,543,744 | 46,339,607 | 35|
| mini_st2stv2-i3d-resnet-50-ts-f64-multisteps-syncbn-bs36-e35 | 69.0973 | 91.7599 | 235,681,087,488 | 46,339,607 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-inception-v1-f8-cosine-bs72-e75 | 59.7424 | 85.948 | 12,047,006,720 | 5,706,503 | 75 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e35 | 63.9376 | 89.0245 | 24,094,013,440 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e35 | 65.3279 | 89.1628 | 48,188,026,880 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e35 | 67.5898 | 90.9637 | 96,376,053,760 | 5,706,503 | 35|
| mini_st2stv2-TAM-b3-sum-inception-v1-ts-max-f8-cosine-bs72-e75 | 50.6992 | 78.6423 | 6,570,863,936 | 5,706,503 | 75|
| mini_st2stv2-TAM-b3-sum-inception-v1-ts-max-f16-multisteps-bs72-e35 | 56.1658 | 83.8715 | 13,141,727,872 | 5,706,503 | 35 |
| mini_st2stv2-TAM-b3-sum-inception-v1-ts-max-f32-multisteps-syncbn-bs36-e35 | 60.2101 | 86.9004 | 26,283,455,744 | 5,706,503 | 35|
| mini_st2stv2-TAM-b3-sum-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e35 | 66.3364 | 90.4726 | 52,566,911,488 | 5,706,503 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-resnet-18-f8-cosine-bs72-e75 | 59.1406 | 85.092 | 14,530,768,896 | 11,225,559 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e35 | 62.0561 | 87.6684 | 29,061,537,792 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e35 | 63.1409 | 87.8464 | 58,123,075,584 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e35 | 64.6416 | 89.2984 | 116,246,151,168 | 11,225,559 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-18-ts-max-f8-cosine-bs72-e75 | 48.0973 | 77.3625 | 7,535,155,712 | 11,225,559 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-18-ts-max-f16-multisteps-bs72-e35 | 53.174 | 81.3204 | 15,070,311,424 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-ts-max-f32-multisteps-bs72-e35 | 56.5811 | 84.4902 | 30,140,622,848 | 11,225,559 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-18-ts-max-f64-multisteps-syncbn-bs36-e35 | 63.413 | 88.9764 | 60,281,245,696 | 11,225,559 | 35|
| | | | | |
| mini_st2stv2-TAM-b3-sum-resnet-50-f8-cosine-bs72-e75 | 65.4378 | 89.1432 | 32,831,963,136 | 23,725,655 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-50-f16-multisteps-bs48-e35 | 68.6329 | 90.8467 | 65,663,926,272 | 23,725,655 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e35 | 69.3089 | 91.4464 | 131,327,852,544 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e35 | 71.5532 | 92.5052 | 262,655,705,088 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-ts-max-f8-cosine-bs72-e75 | 54.5216 | 81.4476 | 14,213,054,464 | 23,725,655 | 75 |
| mini_st2stv2-TAM-b3-sum-resnet-50-ts-max-f16-multisteps-bs60-e35 | 59.1067 | 85.7615 | 28,426,108,928 | 23,725,655 | 35 |
| mini_st2stv2-TAM-b3-sum-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e35 | 63.042 | 88.4231 | 56,852,217,856 | 23,725,655 | 35|
| mini_st2stv2-TAM-b3-sum-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e35 | 69.1819 | 91.9292 | 113,704,435,712 | 23,725,655 | 35|

### Dataset:  Mini-Kinetics400

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs |
|------------|-------|-------|-------|--------|--------|
| mini_kinetics400-inception-v1-f8-cosine-bs72-e100 | 70.4178 | 88.9296 | 11,978,817,536 | 5,804,904 | 100|
| mini_kinetics400-inception-v1-f16-multisteps-bs72-e45 | 70.5195 | 88.8991 | 23,957,635,072 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-f32-multisteps-bs48-e45 | 70.4076 | 89.0007 | 47,915,270,144 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-f64-multisteps-syncbn-bs36-e45 | 70.2439 | 89.0447 | 95,830,540,288 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-ts-max-f8-cosine-bs72-e100 | 64.6742 | 85.6664 | 6,521,725,952 | 5,804,904 | 100|
| mini_kinetics400-inception-v1-ts-max-f16-multisteps-bs72-e45 | 67.0326 | 87.1302 | 13,043,451,904 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-ts-max-f32-multisteps-bs72-e45 | 71.0786 | 89.6107 | 26,086,903,808 | 5,804,904 | 45|
| mini_kinetics400-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e45 | 73.3943 | 90.691 | 52,173,807,616 | 5,804,904 | 45|
| | | | | |
| mini_kinetics400-resnet-18-f8-cosine-bs72-e100 | 67.8662 | 87.3437 | 14,508,490,752 | 11,279,112 | 100|
| mini_kinetics400-resnet-18-f16-multisteps-bs72-e45 | 68.4965 | 87.425 | 29,016,981,504 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-f32-multisteps-bs72-e45 | 69.0759 | 88.0655 | 58,033,963,008 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-f64-multisteps-syncbn-bs36-e45 | 68.6281 | 87.7338 | 116,067,926,016 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-ts-f8-cosine-bs72-e100 | 61.6041 | 82.9115 | 7,520,780,288 | 11,279,112 | 100|
| mini_kinetics400-resnet-18-ts-f16-multisteps-bs72-e45 | 64.1862 | 85.4122 | 15,041,560,576 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-ts-f32-multisteps-bs72-e45 | 68.4863 | 88.2891 | 30,083,121,152 | 11,279,112 | 45|
| mini_kinetics400-resnet-18-ts-f64-multisteps-bs48-e45 | 71.0887 | 89.7835 | 60,166,242,304 | 11,279,112 | 45|
| | | | | |
| mini_kinetics400-resnet-50-f8-cosine-bs72-e100 | 72.1358 | 90.1698 | 32,697,090,048 | 23,917,832 | 100|
| mini_kinetics400-resnet-50-f16-multisteps-bs48-e45 | 72.4713 | 89.875 | 65,394,180,096 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-f32-multisteps-syncbn-bs36-e45 | 73.4654 | 90.3963 | 130,788,360,192 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-f64-multisteps-syncbn-bs36-e45 | 73.4511 | 90.3311 | 261,576,720,384 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-ts-max-f8-cosine-bs72-e100 | 66.9411 | 86.5304 | 14,135,984,128 | 23,917,832 | 100|
| mini_kinetics400-resnet-50-ts-max-f16-multisteps-bs72-e45 | 70.1332 | 88.7567 | 28,271,968,256 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e45 | 72.9167 | 90.4573 | 56,543,936,512 | 23,917,832 | 45|
| mini_kinetics400-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e45 | 75.2489 | 91.8648 | 113,087,873,024 | 23,917,832 | 45|
| | | | | |
| | | | | |
| mini_kinetics400-s3d-f8-cosine-bs72-e100 | 67.0428 | 86.7846 | 19,778,626,560 | 8,108,328 | 100|
| mini_kinetics400-s3d-f16-multisteps-bs72-e45 | 69.7062 | 88.8177 | 39,712,547,840 | 8,108,328 | 45|
| mini_kinetics400-s3d-f32-multisteps-syncbn-bs36-e45 | 72.5 | 90.5793 | 79,580,390,400 | 8,108,328 | 45|
| mini_kinetics400-s3d-f64-multisteps-syncbn-bs36-e45 | 74.7309 | 91.5804 | 159,316,075,520 | 8,108,328 | 45|
| mini_kinetics400-s3d-ts-f8-cosine-bs72-e100 | 61.4822 | 83.5722 | 12,264,670,208 | 8,108,328 | 100|
| mini_kinetics400-s3d-ts-f16-multisteps-bs72-e45 | 65.8331 | 86.3576 | 24,529,340,416 | 8,108,328 | 45|
| mini_kinetics400-s3d-ts-f32-multisteps-bs60-e45 | 69.7672 | 88.8686 | 49,058,680,832 | 8,108,328 | 45|
| mini_kinetics400-s3d-ts-max-f64-multisteps-syncbn-bs36-e45 | 73.2012 | 90.8638 | 98,117,361,664 | 8,108,328 | 45|
| | | | | |
| mini_kinetics400-s3d-resnet-18-f8-cosine-bs72-e100 | 67.8357 | 87.4657 | 21,329,215,488 | 15,483,528 | 100|
| mini_kinetics400-s3d-resnet-18-f16-multisteps-bs72-e45 | 70.3365 | 89.5090 | 42,658,430,976 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-f32-multisteps-bs72-e45 | 72.8881 | 90.7899 | 85,316,861,952 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-f64-multisteps-syncbn-bs36-e45 | 74.4512 | 91.6464 | 170,633,723,904 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-ts-f8-cosine-bs72-e100 | 59.571 | 82.3828 | 12,125,732,864 | 15,483,528 | 100|
| mini_kinetics400-s3d-resnet-18-ts-f16-multisteps-bs72-e45 | 63.3222 | 85.1581 | 24,251,465,728 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-ts-f32-multisteps-bs72-e45 | 68.0797 | 87.7097 | 48,502,931,456 | 15,483,528 | 45|
| mini_kinetics400-s3d-resnet-18-ts-f64-multisteps-bs48-e45 | 71.1294 | 90.0579 | 97,005,862,912 | 15,483,528 | 45|
| | | | | |
| mini_kinetics400-s3d-resnet-50-f8-cosine-bs60-e100 | 71.8715 | 89.814 | 39,517,814,784 | 27,716,616 | 100|
| mini_kinetics400-s3d-resnet-50-f16-multisteps-bs48-e45 | 73.6607 | 91.0135 | 79,035,629,568 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-f32-multisteps-syncbn-bs36-e45 | 76.3516 | 92.2764 | 158,071,259,136 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-f64-multisteps-syncbn-bs36-e45 | 77.1582 | 92.8703 | 316,142,518,272 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-ts-f8-cosine-bs72-e100 | 64.4607 | 85.1784 | 19,672,203,264 | 27,716,616 | 100|
| mini_kinetics400-s3d-resnet-50-ts-f16-multisteps-bs72-e45 | 69.0759 | 88.2383 | 39,344,406,528 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-ts-f32-multisteps-syncbn-bs36-e45 | 72.2866 | 90.4167 | 78,688,813,056 | 27,716,616 | 45|
| mini_kinetics400-s3d-resnet-50-ts-f64-multisteps-syncbn-bs36-e45* | 74.9898 | 91.9207 | 157,377,626,112 | 27,716,616 | 43|
| | | | | |
| | | | | |
| mini_kinetics400-i3d-f8-cosine-bs72-e100 | 68.0899 | 87.6182 | 33,573,614,592 | 12,492,264 | 100|
| mini_kinetics400-i3d-f16-multisteps-bs72-e45 | 70.8651 | 89.387 | 67,380,196,352 | 12,492,264 | 45|
| mini_kinetics400-i3d-f32-multisteps-syncbn-bs36-e45 | 73.6687 | 91.1382 | 135,226,327,040 | 12,492,264 | 45|
| mini_kinetics400-i3d-f64-multisteps-syncbn-bs36-e45 | 74.9543 | 91.6921 | 270,452,654,080 | 12,492,264 | 45|
| mini_kinetics400-i3d-ts-max-f8-cosine-bs72-e100 | 62.3869 | 84.0398 | 21,411,353,600 | 12,492,264 | 100|
| mini_kinetics400-i3d-ts-max-f16-multisteps-bs72-e45 | 66.3414 | 86.9472 | 42,822,707,200 | 12,492,264 | 45|
| mini_kinetics400-i3d-ts-max-f32-multisteps-bs72-e45 | 71.1192 | 89.0312 | 85,645,414,400 | 12,492,264 | 45|
| mini_kinetics400-i3d-ts-max-f64-multisteps-syncbn-bs36-e45 | 73.1301 | 91.1281 | 171,290,828,800 | 12,492,264 | 45|
| | | | | |
| mini_kinetics400-i3d-resnet-18-f8-cosine-bs72-e100 | 66.5345 | 86.7948 | 43,217,190,912 | 33,268,872 | 100|
| mini_kinetics400-i3d-resnet-18-f16-multisteps-bs72-e45 | 70.367 | 89.1939 | 86,434,381,824 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-f32-multisteps-bs72-e45 | 72.7559 | 90.3019 | 172,868,763,648 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-f64-multisteps-bs60-e45 | 73.437 | 90.6476 | 345,737,527,296 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-ts-max-f8-cosine-bs72-e100 | 58.6154 | 81.5493 | 22,472,425,472 | 33,268,872 | 100|
| mini_kinetics400-i3d-resnet-18-ts-max-f16-multisteps-bs72-e45 | 63.068 | 84.7616 | 44,944,850,944 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-ts-max-f32-multisteps-bs72-e45 | 66.6972 | 86.9066 | 89,889,701,888 | 33,268,872 | 45|
| mini_kinetics400-i3d-resnet-18-ts-max-f64-multisteps-bs60-e45 | 71.0074 | 89.0922 | 179,779,403,776 | 33,268,872 | 45|
| | | | | |
| mini_kinetics400-i3d-resnet-50-f8-cosine-bs72-e100 | 73.3455 | 90.4442 | 64,180,322,304 | 46,571,144 | 100|
| mini_kinetics400-i3d-resnet-50-f16-multisteps-bs60-e45 | 75.5718 | 91.8573 | 128,360,644,608 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-f32-multisteps-syncbn-bs36-e45 | 77.1646 | 92.6016 | 256,721,289,216 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-f64-multisteps-syncbn-bs36-e45* | 77.6965 | 92.8601 | 513,442,578,432 | 46,571,144 | 41|
| mini_kinetics400-i3d-resnet-50-ts-max-f8-cosine-bs60-e100 | 65.0605 | 85.6155 | 29,460,135,936 | 46,571,144 | 100|
| mini_kinetics400-i3d-resnet-50-ts-max-f16-multisteps-bs60-e45 | 70.0722 | 88.5941 | 58,920,271,872 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e45 | 73.2012 | 90.5996 | 117,840,543,744 | 46,571,144 | 45|
| mini_kinetics400-i3d-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e45 | 75.9191 | 92.0172 | 235,681,087,488 | 46,571,144 | 45|
| | | | | |
| | | | | |
| mini_kinetics400-TAM-b3-sum-inception-v1-f8-cosine-bs72-e100 | 68.7805 | 88.2114 | 12,047,006,720 | 5,822,328 | 100|
| mini_kinetics400-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e45 | 70.0407 | 89.1565 | 24,094,013,440 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e45 | 72.9878 | 91.1585 | 48,188,026,880 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e45 | 74.1057 | 91.5752 | 96,376,053,760 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-ts-max-f8-cosine-bs72-e100 | 58.313 | 81.626 | 6,570,863,936 | 5,822,328 | 100|
| mini_kinetics400-TAM-b3-sum-inception-v1-ts-max-f16-multisteps-bs72-e45 | 61.4431 | 84.4106 | 13,141,727,872 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-ts-max-f32-multisteps-syncbn-bs36-e45 | 66.7276 | 87.5915 | 26,283,455,744 | 5,822,328 | 45|
| mini_kinetics400-TAM-b3-sum-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e45 | 70.3557 | 89.6545 | 52,566,911,488 | 5,822,328 | 45|
| | | | | |
| mini_kinetics400-TAM-b3-sum-resnet-18-f8-cosine-bs72-e100 | 69.0759 | 88.3501 | 14,530,768,896 | 11,283,528 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e45 | 71.25 | 89.502 | 29,061,537,792 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e45 | 72.6118 | 90.3455 | 58,123,075,584 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e45 | 72.937 | 90.7724 | 116,246,151,168 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-ts-max-f8-cosine-bs72-e100 | 59.1847 | 82.2914 | 7,535,155,712 | 11,283,528 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-18-ts-max-f16-multisteps-bs72-e45 | 62.6423 | 84.624 | 15,070,311,424 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-ts-max-f32-multisteps-bs72-e45 | 67.2053 | 87.439 | 30,140,622,848 | 11,283,528 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-18-ts-max-f64-multisteps-syncbn-bs36-e45 | 69.9797 | 89.187 | 60,281,245,696 | 11,283,528 | 45|
| | | | | |
| mini_kinetics400-TAM-b3-sum-resnet-50-f8-cosine-bs72-e100 | 74.065 | 90.3557 | 32,831,963,136 | 23,957,192 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-50-f16-multisteps-bs48-e45 | 76.4329 | 92.0427 | 65,663,926,272 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e45 | 77.8353 | 92.8963 | 131,327,852,544 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e45 | 78.3252 | 93.4105 | 262,655,705,088 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-ts-max-f8-cosine-bs72-e100 | 67.3374 | 87.124 | 14,213,054,464 | 23,957,192 | 100|
| mini_kinetics400-TAM-b3-sum-resnet-50-ts-max-f16-multisteps-bs60-e45 | 70.3963 | 88.9634 | 28,426,108,928 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e45 | 73.3334 | 90.8943 | 56,852,217,856 | 23,957,192 | 45|
| mini_kinetics400-TAM-b3-sum-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e45 | 75.8684 | 92.1694 | 113,704,435,712 | 23,957,192 | 45|




### Dataset:  Mini-Moments

| Model Name | Top-1 | Top-5 | Flops | Params | Epochs |
|------------|-------|-------|-------|--------|--------|
| mini_moments-inception-v1-f8-cosine-bs72-e75 | 21.73 | 44.46 | 11,978,817,536 | 5,804,904 | 75|
| mini_moments-inception-v1-f16-multisteps-bs72-e35 | 23.3 | 46.52 | 23,957,635,072 | 5,804,904 | 35|
| mini_moments-inception-v1-f32-multisteps-bs48-e35 | 23.09 | 46.54 | 47,915,270,144 | 5,804,904 | 35|
| mini_moments-inception-v1-f64-multisteps-syncbn-bs36-e35 | 23.0815 | 46.6427 | 95,830,540,288 | 5,804,904 | 35|
| mini_moments-inception-v1-ts-f8-cosine-bs72-e75 | 22.85 | 46.01 | 6,521,725,952 | 5,804,904 | 75|
| mini_moments-inception-v1-ts-f16-multisteps-bs72-e35 | 24.59 | 49.7 | 13,043,451,904 | 5,804,904 | 35|
| mini_moments-inception-v1-ts-f32-multisteps-bs72-e35 | 25.63 | 51.24 | 26,086,903,808 | 5,804,904 | 35|
| mini_moments-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e35 | 26.1291 | 51.7386 | 52,173,807,616 | 5,804,904 | 35|
| | | | | |
| mini_moments-resnet-18-f8-cosine-bs72-e75 | 20.55 | 40.69 | 14,508,490,752 | 11,279,112 | 75|
| mini_moments-resnet-18-f16-multisteps-bs72-e35 | 21.23 | 42.38 | 29,016,981,504 | 11,279,112 | 35|
| mini_moments-resnet-18-f32-multisteps-bs72-e35 | 20.92 | 41.65 | 58,033,963,008 | 11,279,112 | 35|
| mini_moments-resnet-18-f64-multisteps-syncbn-bs36-e35 | 20.8958 | 42.7514 | 116,067,926,016 | 11,279,112 | 35|
| mini_moments-resnet-18-ts-max-f8-cosine-bs72-e75 | 20.95 | 43.03 | 7,520,780,288 | 11,279,112 | 75|
| mini_moments-resnet-18-ts-max-f16-multisteps-bs72-e35 | 23.48 | 47.44 | 15,041,560,576 | 11,279,112 | 35|
| mini_moments-resnet-18-ts-max-f32-multisteps-bs72-e35 | 23.74 | 47.67 | 30,083,121,152 | 11,279,112 | 35|
| mini_moments-resnet-18-ts-max-f64-multisteps-bs48-e35 | 24.79 | 49.11 | 60,166,242,304 | 11,279,112 | 35|
| | | | | |
| mini_moments-resnet-50-f8-cosine-bs72-e75 | 22.57 | 44.46 | 32,697,090,048 | 23,917,832 | 75|
| mini_moments-resnet-50-f16-multisteps-bs60-e35 | 23.65 | 47.15 | 65,394,180,096 | 23,917,832 | 35|
| mini_moments-resnet-50-f32-multisteps-syncbn-bs36-e35 | 23.8909 | 47.542 | 130,788,360,192 | 23,917,832 | 35|
| mini_moments-resnet-50-f64-multisteps-syncbn-bs36-e35 | 23.8909 | 47.6319 | 261,576,720,384 | 23,917,832 | 35|
| mini_moments-resnet-50-ts-max-f8-cosine-bs72-e75 | 23.48 | 46.33 | 14,135,984,128 | 23,917,832 | 75|
| mini_moments-resnet-50-ts-max-f16-multisteps-bs72-e35 | 25.53 | 50.64 | 28,271,968,256 | 23,917,832 | 35|
| mini_moments-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e35 | 27.1783 | 52.6879 | 56,543,936,512 | 23,917,832 | 35|
| mini_moments-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e35 | 27.7978 | 54.0867 | 113,087,873,024 | 23,917,832 | 35|
| | | | | |
| | | | | |
| mini_moments-s3d-f8-cosine-bs72-e75 | 21.56 | 44.42 | 19,778,626,560 | 8,108,328 | 75|
| mini_moments-s3d-f16-multisteps-bs72-e35 | 24.45 | 49.27 | 39,712,547,840 | 8,108,328 | 35|
| mini_moments-s3d-f32-multisteps-syncbn-bs36-e35 | 25.4097 | 51.0492 | 79,580,390,400 | 8,108,328 | 35|
| mini_moments-s3d-f64-multisteps-syncbn-bs36-e35 | 26.5987 | 52.3082 | 159,316,075,520 | 8,108,328 | 35|
| mini_moments-s3d-ts-f8-cosine-bs72-e75 | 20.53 | 42.37 | 12,264,670,208 | 8,108,328 | 75|
| mini_moments-s3d-ts-f16-multisteps-bs72-e35 | 24.07 | 48.02 | 24,529,340,416 | 8,108,328 | 35|
| mini_moments-s3d-ts-f32-multisteps-bs60-e35 | 25.79 | 51.08 | 49,058,680,832 | 8,108,328 | 35|
| mini_moments-s3d-ts-max-f64-multisteps-syncbn-bs36-e35 | 26.3789 | 52.2782 | 98,117,361,664 | 8,108,328 | 35|
| | | | | |
| mini_moments-s3d-resnet-18-f8-cosine-bs72-e75 | 22.06 | 45.2 | 21,329,215,488 | 15,483,528 | 75|
| mini_moments-s3d-resnet-18-f16-multisteps-bs72-e35 | 24.55 | 49.22 | 42,658,430,976 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-f32-multisteps-bs72-e35 | 25.6 | 50.74 | 85,316,861,952 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-f64-multisteps-syncbn-bs36-e35 | 25.6695 | 51.1791 | 170,633,723,904 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-ts-f8-cosine-bs72-e75 | 19.12 | 40.73 | 12,125,732,864 | 15,483,528 | 75|
| mini_moments-s3d-resnet-18-ts-f16-multisteps-bs72-e35 | 22.14 | 45.71 | 24,251,465,728 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-ts-f32-multisteps-bs72-e35 | 23.73 | 48.41 | 48,502,931,456 | 15,483,528 | 35|
| mini_moments-s3d-resnet-18-ts-f64-multisteps-bs48-e35 | 24.95 | 49.74 | 97,005,862,912 | 15,483,528 | 35|
| | | | | |
| mini_moments-s3d-resnet-50-f8-cosine-bs72-e75 | 24.42 | 48.6 | 39,517,814,784 | 27,716,616 | 75|
| mini_moments-s3d-resnet-50-f16-multisteps-bs48-e35 | 26.99 | 52.72 | 79,035,629,568 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 27.9576 | 53.5971 | 158,071,259,136 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 27.9477 | 54.2965 | 316,142,518,272 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-ts-f8-cosine-bs72-e75 | 21.08 | 44.15 | 19,672,203,264 | 27,716,616 | 75|
| mini_moments-s3d-resnet-50-ts-f16-multisteps-bs60-e35 | 24.55 | 50.1 | 39,344,406,528 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-ts-f32-multisteps-syncbn-bs36-e35 | 26.6287 | 52.2982 | 78,688,813,056 | 27,716,616 | 35|
| mini_moments-s3d-resnet-50-ts-f64-multisteps-syncbn-bs36-e35 | 27.7578 | 53.9568 | 157,377,626,112 | 27,716,616 | 35|
| | | | | |
| | | | | |
| mini_moments-i3d-f8-cosine-bs60-e75 | 22.42 | 45.54 | 33,573,614,592 | 12,492,264 | 75|
| mini_moments-i3d-f16-multisteps-bs60-e35 | 25.56 | 50.27 | 67,380,196,352 | 12,492,264 | 35|
| mini_moments-i3d-f32-multisteps-bs48-e35 | 26.17 | 51.82 | 135,226,327,040 | 12,492,264 | 35|
| mini_moments-i3d-f64-multisteps-syncbn-bs36-e35 | 26.4388 | 52.2782 | 270,452,654,080 | 12,492,264 | 35|
| mini_moments-i3d-ts-max-f8-cosine-bs72-e75 | 21.83 | 44.08 | 21,411,353,600 | 12,492,264 | 75|
| mini_moments-i3d-ts-max-f16-multisteps-bs72-e35 | 24.66 | 48.75 | 42,822,707,200 | 12,492,264 | 35|
| mini_moments-i3d-ts-max-f32-multisteps-bs72-e35 | 25.94 | 51.47 | 85,645,414,400 | 12,492,264 | 35|
| mini_moments-i3d-ts-max-f64-multisteps-syncbn-bs36-e35 | 26.6687 | 52.5979 | 171,290,828,800 | 12,492,264 | 35|
| | | | | |
| mini_moments-i3d-resnet-18-f8-cosine-bs72-e75 | 20.9 | 41.76 | 43,217,190,912 | 33,268,872 | 75|
| mini_moments-i3d-resnet-18-f16-multisteps-bs72-e35 | 22.31 | 45.11 | 86,434,381,824 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-f32-multisteps-bs72-e35 | 22.66 | 45.71 | 172,868,763,648 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-f64-multisteps-bs60-e35 | 22.7 | 45.94 | 345,737,527,296 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-ts-max-f8-cosine-bs72-e75 | 19.35 | 39.83 | 22,472,425,472 | 33,268,872 | 75|
| mini_moments-i3d-resnet-18-ts-max-f16-multisteps-bs72-e35 | 21.66 | 43.81 | 44,944,850,944 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-ts-max-f32-multisteps-bs72-e35 | 22.32 | 45.8 | 89,889,701,888 | 33,268,872 | 35|
| mini_moments-i3d-resnet-18-ts-max-f64-multisteps-bs60-e35 | 23.36 | 46.01 | 179,779,403,776 | 33,268,872 | 35|
| | | | | |
| mini_moments-i3d-resnet-50-f8-cosine-bs72-e75 | 24.57 | 48.1 | 64,180,322,304 | 46,571,144 | 75|
| mini_moments-i3d-resnet-50-f16-multisteps-bs60-e35 | 26.51 | 52.03 | 128,360,644,608 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-f32-multisteps-syncbn-bs36-e35 | 27.498 | 53.4672 | 256,721,289,216 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-f64-multisteps-syncbn-bs36-e35 | 27.6279 | 53.2774 | 513,442,578,432 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-ts-max-f8-cosine-bs60-e75 | 22.64 | 44.34 | 29,460,135,936 | 46,571,144 | 75|
| mini_moments-i3d-resnet-50-ts-max-f16-multisteps-bs60-e35 | 25.41 | 50.51 | 58,920,271,872 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-ts-f32-multisteps-syncbn-bs36-e35 | 27.1346 | 52.9494 | 117,840,543,744 | 46,571,144 | 35|
| mini_moments-i3d-resnet-50-ts-f64-multisteps-syncbn-bs36-e35 | 27.8377 | 53.9768 | 235,681,087,488 | 46,571,144 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-inception-v1-f8-cosine-bs72-e75 | 23.2953 | 47.2905 | 12,047,006,720 | 5,822,328 | 75 |
| mini_moments-TAM-b3-sum-inception-v1-f16-multisteps-bs72-e35 | 25.9148 | 50.9398 | 24,094,013,440 | 5,822,328 | 35 |
| mini_moments-TAM-b3-sum-inception-v1-f32-multisteps-syncbn-bs36-e35 | 26.3247 | 52.1395 | 48,188,026,880 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e35 | 26.7986 | 53.0975 | 96,376,053,760 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-ts-max-f8-cosine-bs72-e75 | 21.4757 | 44.3111 | 6,570,863,936 | 5,822,328 | 75 |
| mini_moments-TAM-b3-sum-inception-v1-ts-max-f16-multisteps-bs72-e35 | 23.8852 | 48.1404 | 13,141,727,872 | 5,822,328 | 35 |
| mini_moments-TAM-b3-sum-inception-v1-ts-max-f32-multisteps-syncbn-bs36-e35 | 25.6449 | 50.9798 | 26,283,455,744 | 5,822,328 | 35|
| mini_moments-TAM-b3-sum-inception-v1-ts-max-f64-multisteps-syncbn-bs36-e35 | 26.5687 | 52.6379 | 52,566,911,488 | 5,822,328 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-resnet-18-f8-cosine-bs72-e75 | 22.0556 | 44.881 | 14,530,768,896 | 11,283,528 | 75 |
| mini_moments-TAM-b3-sum-resnet-18-f16-multisteps-bs72-e35 | 24.1252 | 48.3003 | 29,061,537,792 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-f32-multisteps-bs72-e35 | 24.4451 | 48.8702 | 58,123,075,584 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-f64-multisteps-syncbn-bs36-e35 | 24.3751 | 49.74 | 116,246,151,168 | 11,283,528 | 35|
| mini_moments-TAM-b3-sum-resnet-18-ts-max-f8-cosine-bs72-e75 | 21.58 | 44.92 | 7,535,155,712 | 11,283,528 | 75 |
| mini_moments-TAM-b3-sum-resnet-18-ts-max-f16-multisteps-bs72-e35 | 23.79 | 48.02 | 15,070,311,424 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-ts-max-f32-multisteps-bs72-e35 | 24.775 | 49.4001 | 30,140,622,848 | 11,283,528 | 35 |
| mini_moments-TAM-b3-sum-resnet-18-ts-max-f64-multisteps-syncbn-bs36-e35 | 25.4949 | 51.3497 | 60,281,245,696 | 11,283,528 | 35|
| | | | | |
| mini_moments-TAM-b3-sum-resnet-50-f8-cosine-bs72-e75 | 25.9748 | 50.08 | 32,831,963,136 | 23,957,192 | 75 |
| mini_moments-TAM-b3-sum-resnet-50-f16-multisteps-bs36-e35 | 28.2344 | 54.809 | 65,663,926,272 | 23,957,192 | 35 |
| mini_moments-TAM-b3-sum-resnet-50-f32-multisteps-syncbn-bs36-e35 | 28.8269 | 55.6055 | 131,327,852,544 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e35 | 28.8569 | 55.2658 | 262,655,705,088 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-ts-max-f8-cosine-bs72-e75 | 24.8 | 48.85 | 14,213,054,464 | 23,957,192 | 75 |
| mini_moments-TAM-b3-sum-resnet-50-ts-max-f16-multisteps-bs60-e35 | 26.9446 | 52.4495 | 28,426,108,928 | 23,957,192 | 35 |
| mini_moments-TAM-b3-sum-resnet-50-ts-max-f32-multisteps-syncbn-bs36-e35 | 28.6371 | 55.1259 | 56,852,217,856 | 23,957,192 | 35|
| mini_moments-TAM-b3-sum-resnet-50-ts-max-f64-multisteps-syncbn-bs36-e35 | 29.2766 | 55.3957 | 113,704,435,712 | 23,957,192 | 35|
