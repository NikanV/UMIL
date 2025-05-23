# Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n FrameShield python=3.10
conda activate FrameShield
pip install -r requirements.txt
```

> There might be some issues with pip dependencies sometimes, in those situations follow these steps instead:

```
pip install safetensors==0.4.5 mmcv-full==1.7.2
pip install torch
pip install -r requirements.txt
```

# Data Preparation

The dataset directories can be in any format you like (as long as you create the proper text files for them), but for the best performance, you can organize them like this one:
```
TAD/
├─ frames/    
    ├─ abnormal/
        ├─ 01_Accident_001.mp4/
        ...
    ├─ normal/ 
        ...
```

## Dataset Text Files

You can change the way that we handle text files in code, but for now they are formatted as below. Also you can find the text files that we used in `labels` directory. 

- train.txt (each row)
    
    ```
    frame-dir-path start_idx end_idx video_label 
    ```
    > start_idx and end_idx are basically 0 and total_frames respectively in most of the train text files.
- test.txt (each row)

    ```
    frame-dir-path total_frames abn_start_idx abn_end_idx video_label 
    ```
    > abn_start_idx and abn_end_idx are the indices where the abnormal frames begin and end respectively in the video.

# Train

The config files lie in `configs`. You can change them based on your dataset, or your desired num of clips, etc.

### Phase1: X-ClipMIL training
```
CUDA_VISIBLE_DEVICES=0 bash tools/server_train.sh 1
```

### Generating Pseudo Labels
```
CUDA_VISIBLE_DEVICES=0 bash tools/server_genlabels.sh 1
```

### Phase2: Asversarial Training
```
CUDA_VISIBLE_DEVICES=0 bash tools/server_advtrain.sh 1
```

# Test

Again config files lie in `configs` and you can change them the way you want.

### Clean test
```
CUDA_VISIBLE_DEVICES=0 bash tools/server_test.sh 1
```

### Adversarial Attack

```
CUDA_VISIBLE_DEVICES=0 bash tools/server_attack.sh 1
```
