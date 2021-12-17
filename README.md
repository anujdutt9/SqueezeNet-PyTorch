# SqueezeNet PyTorch Implementation

This repository is an attempt at implementing SqueezeNet paper using PyTorch.

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
