# SqueezeNet PyTorch Implementation

This repository is an attempt at implementing the paper "[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)" using PyTorch.

# Setup

Please setup your training environemnt by installing the requirements using:

```
$ pip install -r requirements.txt
```

# Training

To run the model training, use the following command:

```
$ python train.py -bs 32 -epochs 100 -lr 0.001 -o ./assets/models/
```

This will train the model and save the model's weights as `state_dict()` to the `assets/models/` folder.
