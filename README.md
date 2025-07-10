# Efficient Motion Prediction (EMP)
### [[Paper]](https://arxiv.org/abs/2409.16154)
> [**Efficient Motion Prediction: A Lightweight & Accurate Trajectory Prediction Model With Fast Training and Inference Speed**](https://arxiv.org/abs/2409.16154)            
> Alexander Prutsch, Horst Bischof, Horst Possegger
> **Graz University of Technology**  
> **IROS 2024**

## Abstract
For efficient and safe autonomous driving, it is essential that autonomous vehicles can predict the motion of other traffic agents.
While highly accurate, current motion prediction models often impose significant challenges in terms of training resource requirements and deployment on embedded hardware.
We propose a new efficient motion prediction model, which achieves highly competitive benchmark results while training only a few hours on a single GPU.
Due to our lightweight architectural choices and the focus on reducing the required training resources, our model can easily be applied to custom datasets.
Furthermore, its low inference latency makes it particularly suitable for deployment in autonomous applications with limited computing resources.

## Getting Started

### Create and Activate Virtual Environment
```
conda create -n emp python=3.8.18
conda activate emp
```

### Install PyTorch
We tested our implementation with torch 1.11.0+cu113 and torch 2.1.1+cu121.

Install PyTorch e.g.
```
pip --no-cache-dir install torch==1.11.0+cu113  torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install Dependencies
```
pip install -r ./requirements.txt
pip install av2
```

### Download Download [Argoverse 2 Motion Forecasting Dataset](https://argoverse.github.io/user-guide/datasets/motion_forecasting.html#download)
The expected structure of the AV2 data should be:
```
data_root
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### Data Preprocessing
Preprocess the Argoverse 2 dataset by executing
```
python preprocess.py --data_root=/path/to/data_root -p
```

## Training
Train EMP-M/D model using
```
python train.py data_root=/path/to/data_root model=emp gpus=1 batch_size=96 monitor=val_minFDE6 model.target.decoder=mlp
```
Use `model.target.decoder=mlp` for EMP-M and `model.target.decoder=detr` for EMP-D.

## Evaluation
Run evaluation using
```
python eval.py data_root=/path/to/data_root batch_size=32 'checkpoint="/path/to/checkpoint.ckpt"'
```

## Evaluation
To visualize scenario data and model predictions use
```
python visualize.py -p
```
Please set the datafolder, split and checkpoint directly in the `visualize.py` script.

Without `-p`, only the input data is visualized.

## Pretrained Model Weights
You can find our pretrained weights for AV2 in the checkpoints folder.
- [EMP-M](checkpoints/empm.ckpt)
- [EMP-D](checkpoints/empd.ckpt)

## Bibtex
```bibtex
@inproceedings{prutsch2024efficient,
 title={{Efficient Motion Prediction: A Lightweight & Accurate Trajectory Prediction Model With Fast Training and Inference Speed}},
 author={Alexander Prutsch, Horst Bischof, Horst Possegger},
 booktitle={IROS},
 year={2024},
}
```

## Acknowledgements
This repository is based on [Forecast-MAE](https://github.com/jchengai/forecast-mae). We thank them for their excellent work!