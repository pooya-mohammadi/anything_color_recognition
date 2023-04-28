# Anything Color Recognition

In this repository, the codes for training anything color recognition are provided. This is a pytorch implementation of following paper:
https://arxiv.org/pdf/1510.07391.pdf named **Vehicle Color Recognition using Convolutional Neural Network**


![img.png](img.png)

The colors are as follows:
[`Black`, `Blue`, `Brown`, `Green`, `Pink`, `Red`, `Silver`, `White`, `Yellow`].

In this repo I try to train a model to recognize the color of anything. Right now the model is trained on
vehicle dataset and fabric dataset. I will be adding more objects to the dataset.

# Dataset
The dataset is mainly based on vehicle dataset. The rest are also being added to the dataset. 

# Installation:
```commandline
pip install -r requirements.txt
```

# Hyper parameters
The hyper-parameters are set in the `settings.py` module. You can adjust them based on your need. 

# Train
```commandline
python train.py
```

# Graphs
```commandline
cd output/exp_4/finetune/lightning/version_0/
tensorboard --logdir .
```

# inference.py
```commandline
python inference.py --model_path output/exp_4/best.ckpt --labels_map_path output/exp_4/labels_map.pkl --img_path sample_images/sample_01.jpg
```

# References:
1. https://arxiv.org/pdf/1510.07391.pdf
2. https://github.com/pooya-mohammadi/deep_utils