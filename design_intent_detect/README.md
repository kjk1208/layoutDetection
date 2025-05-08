# DensityLayout: Density-Conditioned Layout GAN for Visual-Textual Presentation Designs

This subrepository contains the Pytorch implementation for the design intent detection model, which is first preseneted in "[Density-Conditioned Layout GAN for Visual-Textual Presentation Designs](https://link.springer.com/chapter/10.1007/978-3-031-46308-2_16)", ICIG 2023.
In PosterO [CVPR 2025], we re-implement the model using the [UNet with a MiT-B1 encoder](https://smp.readthedocs.io/en/latest/encoders.html).

# Train Design Intent Detection Model From Scratch

Before we start, make sure you have specified AbsolutePath/to/DatasetDirectory in ```init_path.sh``` and execute the following command:
```
source init_path.sh
```

1. Data Preparation
- Execute the following commands to obtain the binary masks of layout element regions, which are used for weakly-supervised learning:
```
cd design_intent_detect
source pp.sh
```

2. Training the Model
- Specify the GPU IDs in the ```train.sh``` file and execute the following command:
```
source train.sh <dataset>
```
For example, ```source train.sh pku``` for the PKU PosterLayout dataset.

3. Testing the Model
- Specify the GPU IDs in the ```test.sh``` file and execute the following command:
```
cd design_intent_detect
source test.sh <DATASET> <PATH_TO_WEIGHT>
```
- With the default setting, the detection results will be saved under ```<dataset>_128_1e-06_none/result```, including
    - maps: results will be saved under ```<dataset>_128_1e-06_none/result/<split>```
    - features: results will be saved under ```<dataset>_128_1e-06_none/result/<dataset>_features```
    - boxes (available areas): results will be saved as ```<dataset>_128_1e-06_none/result/design_intent_bbox_<split>.pt```