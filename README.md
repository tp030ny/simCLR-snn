# SimCLR-snn-version
PyTorch implementation of SimCLR by snn: A Simple Framework for Contrastive Learning of Visual Representations by T. Chen et al.
Including support for:
- Distributed data parallel training
- Global batch normalization
[Link to paper](https://arxiv.org/pdf/2002.05709.pdf)


### Training ResNet encoder:
Simply run the following to pre-train a ResNet encoder using SimCLR_snn version on the CIFAR-10 dataset:
```
python main.py --dataset CIFAR10 --resnet="resnet32" --spiking=True
```

### Distributed Training
With distributed data parallel (DDP) training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

#### LARS optimizer
The LARS optimizer is implemented in `modules/lars.py`. It can be activated by adjusting the `config/config.yaml` optimizer setting to: `optimizer: "LARS"`. It is still experimental and has not been thoroughly tested.

## What is SimCLR?
SimCLR is a "simple framework for contrastive learning of visual representations". The contrastive prediction task is defined on pairs of augmented examples, resulting in 2N examples per minibatch. Two augmented versions of an image are considered as a correlated, "positive" pair (x_i and x_j). The remaining 2(N - 1) augmented examples are considered negative examples. The contrastive prediction task aims to identify x_j in the set of negative examples for a given x_i.

<p align="center">
  <img src="https://github.com/Spijkervet/SimCLR/blob/master/media/architecture.png?raw=true" width="500"/>
</p>

## Usage
Run the following command to setup a conda environment:
```
sh setup.sh
conda activate simclr
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run for single GPU or CPU training:
```
python main.py --resnet="resnet32" --spiking=True
```

For distributed training (DDP), use for every process in nodes, in which N is the GPU number you would like to dedicate the process to:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

`--nr` corresponds to the process number of the N nodes we make available for training.

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `40`).

```
python linear_evaluation.py
```

or in place:
```
python linear_evaluation.py --model_path=./save --epoch_num=100 --resnet="resnet32" --spiking=True
```


## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. An example `config.yaml` file:
```
# train options
batch_size: 256
workers: 16
start_epoch: 0
epochs: 40
dataset_dir: "./datasets"

# model options
resnet: "resnet18"
normalize: True
projection_dim: 64

# loss options
temperature: 0.5

# reload options
model_path: "logs/0" # set to the directory containing `checkpoint_##.tar` 
epoch_num: 40 # set to checkpoint number

# logistic regression options
logistic_batch_size: 256
logistic_epochs: 100
```


#### Dependencies
```
torch
torchvision
tensorboard
pyyaml
```
