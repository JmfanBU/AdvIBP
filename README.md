# Adversarial Training and Provable Robustness: A Tale of Two Objectives

*AdvIBP* is the official implementation of Adversarial Training and Provable Robustness: A Tale of Two Objectives. *AdvIBP* is a certifed adversarial training method that combines adversarial training and provable robustness verification. *AdvIBP* matches or outperforms state-of-art approach, [CROWN-IBP](https://openreview.net/pdf?id=Skxuk1rFwB), for provable L\_infinity robustness on MNIST and CIFAR. The detailed comparison is shown in Table 1.

Our repository provides high quality PyTorch implementations of *AdvIBP*. We implemented Multi-GPU training for training large models.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Training large models requires multiple GPUs.

## Training
Our program is tested on Pytorch 1.4.0 and Python 3.6.9.

We have all training parameters included in JSON files, under the config directory. We provide configuration files which can reproduce the results in our paper.

To train *AdvIBP* on MNIST for small and medium models, run:

```bash
# train under test epsilon=0.1
python train.py --config mnist_IBP_Adv_01_02.json
# train under test epsilon=0.3
python train.py --config mnist_IBP_Adv.json
```

To train *AdvIBP* on MNIST for large model, run:

```bash
# This uses 4 GPUs by default.
# train under test epsilon=0.1
python train.py --config mnist_IBP_Adv_Large_01_02.json
# train under test epsilon=0.3
python train.py --config mnist_IBP_Adv_Large.json
```

To train *AdvIBP* on CIFAR-10 for small and medium models, run:

```bash
# train under test epsilon=2/255
python train.py --config cifar_IBP_Adv_2_255.json
# train under test epsilon=8/255
python train.py --config cifar_IBP_Adv.json
# train under test epsilon=16/255
python train.py --config cifar_IBP_Adv_16_255.json
```

To train *AdvIBP* on MNIST for large model, run:

```bash
# This uses 4 GPUs by default.
# train under test epsilon=2/255
python train.py --config cifar_IBP_Adv_Large_2_255.json
# train under test epsilon=8/255
python train.py --config cifar_IBP_Adv_Large.json
# train under test epsilon=16/255
python train.py --config cifar_IBP_Adv_Large_16_255.json
```

## Pre-trained Models

You can download pretrained models here:

- [trained_IBP_models](https://drive.google.com/mymodel.pth): Training by *AdvIBP* on MNIST and CIFAR-10 for different model architectures and different perturbation settings. 

To evaluate the best (and largest) MNIST and CIFAR-10 model (same model structure as in Gowal et al. 2018, referred to as "dm-large" in our paper), run:

```bash
# Evaluate large model on MNIST with epsilon=0.1
python train.py --config eval/mnist_IBP_Adv_Large_01_eval.json
# Evaluate large model on MNIST with epsilon=0.2
python train.py --config eval/mnist_IBP_Adv_Large_02_eval.json
# Evaluate large model on MNIST with epsilon=0.3
python train.py --config eval/mnist_IBP_Adv_Large_03_eval.json
# Evaluate large model on MNIST with epsilon=0.4
python train.py --config eval/mnist_IBP_Adv_Large_04_eval.json

# Evaluate large model on CIFAR-10 with epsilon=2/255
python train.py --config eval/cifar_IBP_Adv_Large_2_255_eval.json
# Evaluate large model on CIFAR-10 with epsilon=8/255
python train.py --config eval/cifar_IBP_Adv_Large_8_255_eval.json
# Evaluate large model on CIFAR-10 with epsilon=16/255
python train.py --config eval/cifar_IBP_Adv_Large_16_255_eval.json
```

To evaluate the medium MNIST and CIFAR-10 model (same model structure as in Gowal et al. 2018, referred to as "dm-medium" in our paper), run:
```bash
# Evaluate medium model on MNIST with epsilon=0.1
python train.py --config eval/mnist_IBP_Adv_Med_01_eval.json
# Evaluate medium model on MNIST with epsilon=0.2
python train.py --config eval/mnist_IBP_Adv_Med_02_eval.json
# Evaluate medium model on MNIST with epsilon=0.3
python train.py --config eval/mnist_IBP_Adv_Med_03_eval.json
# Evaluate medium model on MNIST with epsilon=0.4
python train.py --config eval/mnist_IBP_Adv_Med_04_eval.json

# Evaluate medium model on CIFAR-10 with epsilon=2/255
python train.py --config eval/cifar_IBP_Adv_Med_2_255_eval.json
# Evaluate medium model on CIFAR-10 with epsilon=8/255
python train.py --config eval/cifar_IBP_Adv_Med_8_255_eval.json
# Evaluate medium model on CIFAR-10 with epsilon=16/255
python train.py --config eval/cifar_IBP_Adv_Med_16_255_eval.json
```


To evaluate the small MNIST and CIFAR-10 model (same model structure as in Gowal et al. 2018, referred to as "dm-small" in our paper), run:

```bash
# Evaluate small model on MNIST with epsilon=0.1
python train.py --config eval/mnist_IBP_Adv_Small_01_eval.json
# Evaluate small model on MNIST with epsilon=0.2
python train.py --config eval/mnist_IBP_Adv_Small_02_eval.json
# Evaluate small model on MNIST with epsilon=0.3
python train.py --config eval/mnist_IBP_Adv_Small_03_eval.json
# Evaluate small model on MNIST with epsilon=0.4
python train.py --config eval/mnist_IBP_Adv_Small_04_eval.json

# Evaluate small model on CIFAR-10 with epsilon=2/255
python train.py --config eval/cifar_IBP_Adv_Small_2_255_eval.json
# Evaluate small model on CIFAR-10 with epsilon=8/255
python train.py --config eval/cifar_IBP_Adv_Small_8_255_eval.json
# Evaluate small model on CIFAR-10 with epsilon=16/255
python train.py --config eval/cifar_IBP_Adv_Small_16_255_eval.json
```


## Results

Our model achieves the following standard, verified and PGD errors compared with CROWN-IBP:
<p align="center">
    <img src="figures/result_table.png" alt>
</p>
<p align="center">
    <em>Table 1: Comparison with CROWN-IBP</em>
</p>


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
