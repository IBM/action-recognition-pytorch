## Experimental Setting
- All trained from 8-frame model (finetune from ImageNet-pretrained), while more frame models are finetuned from fewer-frame model. 
- Number of frames (`-fX`, `X` denotes frame number.)
- Backbone (2D or 3D)
- With or without temporal pooling in the model (`-tp` means with temporal pooling)


## Performance

The model name should self-explain the experimental setting

### Dataset:  Something-Something-V2

| Model Name | Top-1 | Top-5 | Flops | Params |Epochs|
|------------|-------|-------|-------|--------|-------|
| st2stv2-TAM-b3-sum-inception-v1-f16-cosine-bs72-e75 | 58.632 | 86.1582 | 24,094,013,440 | 5,795,678 | 75|
| st2stv2-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e35 | 60.347 | 87.7442 | 96,376,053,760 | 5,795,678 | 35|
| st2stv2-TAM-b3-sum-resnet-50-f16-cosine-bs48-e75 | 61.929 | 88.2324 | 65,663,926,272 | 23,903,918 | 75|
| st2stv2-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e35 | 64.5965 | 90.2865 | 262,655,705,088 | 23,903,918 | 35|
| st2stv2-i3d-resnet-50-tp-f16-cosine-bs72-e75 | 52.6254 | 80.385 | 58,920,271,872 | 46,517,870 | 75|
| st2stv2-i3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e35 | 64.0839 | 89.8103 | 235,681,087,488 | 46,517,870 | 35|
| st2stv2-inception-v1-f16-cosine-bs72-e75 | 30.6413 | 62.1262 | 23,957,635,072 | 5,778,254 | 75|
| st2stv2-inception-v1-f64-multisteps-syncbn-bs36-e35 | 24.7014 | 53.7974 | 95,830,540,288 | 5,778,254 | 35|
| st2stv2-resnet-50-f16-cosine-bs60-e75 | 30.7099 | 61.5894 | 65,394,180,096 | 23,864,558 | 75|
| st2stv2-resnet-50-f64-multisteps-syncbn-bs36-e35 | 29.4319 | 60.5907 | 261,576,720,384 | 23,864,558 | 35|
| st2stv2-s3d-tp-max-f16-cosine-bs72-e75 | 52.6981 | 81.4061 | 24,529,340,416 | 8,081,678 | 75|
| st2stv2-s3d-tp-max-f64-multisteps-syncbn-bs36-e35 | 61.8563 | 88.8055 | 98,117,361,664 | 8,081,678 | 35|

### Dataset:  Kinetics400

| Model Name | Top-1 | Top-5 | Flops | Params |Epochs|
|------------|-------|-------|-------|--------|-------|
| kinetics400-TAM-b3-sum-inception-v1-f16-cosine-bs72-e100 | 66.6921 | 87.5673 | 24,094,013,440 | 6,027,328 | 100|
| kinetics400-TAM-b3-sum-inception-v1-f64-multisteps-syncbn-bs36-e45 | 68.4857 | 88.7551 | 96,376,053,760 | 6,027,328 | 45|
| kinetics400-TAM-b3-sum-resnet-50-f16-cosine-bs48-e100 | 68.3847 | 88.2383 | 65,663,926,272 | 24,366,992 | 100|
| kinetics400-TAM-b3-sum-resnet-50-f64-multisteps-syncbn-bs36-e45 | 73.9736 | 91.5803 | 262,655,705,088 | 24,366,992 | 45|
| kinetics400-i3d-resnet-50-tp-f16-cosine-bs72-e100 | 67.1208 | 86.8524 | 58,920,271,872 | 46,980,944 | 100|
| kinetics400-i3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e45 | 72.1392 | 90.4878 | 235,681,087,488 | 46,980,944 | 45|
| kinetics400-i3d-tp-max-f16-cosine-bs72-e100 | 63.1959 | 84.8442 | 42,822,707,200 | 12,697,264 | 100|
| kinetics400-i3d-tp-max-f64-multisteps-syncbn-bs36-e45 | 68.0335 | 88.5061 | 171,290,828,800 | 12,697,264 | 45|
| kinetics400-inception-v1-f16-cosine-bs72-e100 | 65.1889 | 86.3999 | 23,957,635,072 | 6,009,904 | 100|
| kinetics400-inception-v1-f64-multisteps-syncbn-bs36-e45 | 64.2879 | 85.524 | 95,830,540,288 | 6,009,904 | 45|
| kinetics400-resnet-50-f16-cosine-bs48-e75 | 67.3191 | 87.6099 | 65,394,180,096 | 24,327,632 | 75|
| kinetics400-resnet-50-f64-multisteps-syncbn-bs36-e45 | 67.9324 | 88.0605 | 261,576,720,384 | 24,327,632 | 45|
| kinetics400-s3d-resnet-50-tp-f16-cosine-bs66-e100 | 65.8447 | 86.4253 | 39,344,406,528 | 28,126,416 | 100|
| kinetics400-s3d-resnet-50-tp-f64-multisteps-syncbn-bs36-e45 | 70.8651 | 89.9005 | 157,377,626,112 | 28,126,416 | 45|
| kinetics400-s3d-tp-max-f16-cosine-bs72-e100 | 62.8451 | 84.956 | 24,529,340,416 | 8,313,328 | 100|
| kinetics400-s3d-tp-max-f64-multisteps-syncbn-bs36-e45 | 68.2469 | 88.6027 | 98,117,361,664 | 8,313,328 | 45|
