# Adversarial Training and Provable Robustness: A Tale of Two Objectives

This repository is the official implementation of Adversarial Training and Provable Robustness: A Tale of Two Objectives. *AdvIBP* is a certifed adversarial training method that combines adversarial training and provable robustness verification. *AdvIBP* matches or outperforms state-of-art approach, [CROWN-IBP](https://openreview.net/pdf?id=Skxuk1rFwB), for provable L\_infinity robustness on MNIST and CIFAR. We achieved state-of-the-art *verified* (certified) error on MNIST and CIFAR: for MNIST, **6.60\%** at `epsilon=0.3` and **12.30\%** at `epsilon=0.4` (L\_infinity norm distortion); and for CIFAR, **66.57\%** at `epsilon=8/255` and **76.05\%** at `epsilon=16/255`.

Jiameng Fan and Wenchao Li, ["Adversarial Training and Provable Robustness: A Tale of Two Objectives"](https://arxiv.org/pdf/2008.06081.pdf), AAAI 2021

Our repository provides high quality PyTorch implementations of *AdvIBP* and *AdvCROWN-IBP*. We implemented Multi-GPU training for training large models.


## Results

### Standard, verified and PGD-200 test errors of the state-of-art models trained with our joint training framework:

| Dataset  | Test epsilon | Model path in trained_IBP_models   | Standard error | Verified error | PGD error |
|----------|--------------|------------------------------------|----------------|----------------|-----------|
| MNIST    | 0.1          | mnist/mnist_large_01_02            | 1.22%          | 2.19%          | 1.57%     |
| MNIST    | 0.2          | mnist/mnist_large_02_04            | 1.51%          | 3.87%          | 1.98%     |
| MNIST    | 0.3          | mnist/mnist_large_03_04            | 1.90%          | 6.60%          | 2.87%     |
| MNIST    | 0.4          | mnist/mnist_large_04_04            | 1.90%          | 12.30%         | 3.46%     |
| CIFAR-10 | 2/255        | cifar-10/cifar_large_2_255         | 40.61%         | 51.66%         | 46.97%    |
| CIFAR-10 | 8/255        | cifar-10/cifar_large_8_255         | 52.86%         | 66.57%         | 61.66%    |
| CIFAR-10 | 16/255       | cifar-10/cifar_large_16_255        | 64.40%         | 76.05%         | 71.78%    |

## Starting with the Code

To install requirements and *AdvIBP*:

```setup
pip install -e .
```

Training large models requires multiple GPUs.

## Training
Our program is tested on Pytorch 1.4.0 and Python 3.6.9.

We have all training parameters included in JSON files, under the config directory. We provide configuration files which can reproduce the results in our paper.

To train *AdvIBP* and *AdvCROWN-IBP* on MNIST for small and medium models, run:

```bash
# train under test epsilon=0.1
python train.py --config mnist_IBP_Adv_01_02.json
python train.py --config mnist_AdvCROWN-IBP_01_02.json
# train under test epsilon=0.3
python train.py --config mnist_IBP_Adv.json
python train.py --config mnist_AdvCROWN-IBP.json
```

To train *AdvIBP* and *AdvCROWN-IBP* on MNIST for large model, run:

```bash
# This uses 4 GPUs by default.
# train under test epsilon=0.1
python train.py --config mnist_IBP_Adv_Large_01_02.json
python train.py --config mnist_AdvCROWN-IBP_Large_01_02.json
# train under test epsilon=0.3
python train.py --config mnist_IBP_Adv_Large.json
python train.py --config mnist_AdvCROWN-IBP_Large.json
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

To train *AdvIBP* on CIFAR-10 for large model, run:

```bash
# This uses 4 GPUs by default.
# train under test epsilon=2/255
python train.py --config cifar_IBP_Adv_Large_2_255.json
# train under test epsilon=8/255
python train.py --config cifar_IBP_Adv_Large.json
# train under test epsilon=16/255
python train.py --config cifar_IBP_Adv_Large_16_255.json
```

## Pre-trained State-of-Art Models

You can download pretrained models here:

- [trained_IBP_models](https://drive.google.com/drive/folders/10R3_1lPciXgHSMivrdwQtF3Vhom9dPiw?usp=sharing): Training by *AdvIBP* and *AdvCROWN-IBP* on MNIST and CIFAR-10 for different model architectures and different perturbation settings. The downloaded models should be in the same directory level as AdvIBP folder. The following command can reproduce the results in Table 2, 3 and 4 of the paper.

To evaluate the state-of-art (and largest) MNIST and CIFAR-10 models (the same model structure as in Gowal et al. 2018, referred to as "dm-large" in our paper), run:

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

To evaluate the medium MNIST and CIFAR-10 models (the same model structure as in Gowal et al. 2018, referred to as "dm-medium" in our paper), run:
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


To evaluate the small MNIST and CIFAR-10 models (the same model structure as in Gowal et al. 2018, referred to as "dm-small" in our paper), run:

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


To evaluate *AdvIBP* and CROWN-IBP on a wide range of models (Table 3), run:
```bash
# Evaluate 10 models on MNIST with epsilon=0.2
python train.py --config eval/mnist_AdvIBP_02_eval.json
python train.py --config eval/mnist_CROWN-IBP_02_eval.json
# Evaluate 10 model on MNIST with epsilon=0.3
python train.py --config eval/mnist_AdvIBP_03_eval.json
python train.py --config eval/mnist_CROWN-IBP_03_eval.json

# Evaluate 7 models on CIFAR-10 with epsilon=8/255
python train.py --config eval/cifar_AdvIBP_eval.json
python train.py --config eval/cifar_CROWN-IBP_eval.json
```


To evaluate *AdvIBP* on three new model structures in Table A and
reproduce the result in Table C

```bash
# Evaluate on MNIST with epsilon=0.1
python train.py --config eval/mnist_new_models_AdvIBP_01_eval.json
# Evaluate on MNIST with epsilon=0.2
python train.py --config eval/mnist_new_models_AdvIBP_02_eval.json
# Evaluate on MNIST with epsilon=0.3
python train.py --config eval/mnist_new_models_AdvIBP_03_eval.json
# Evaluate on MNIST with epsilon=0.4
python train.py --config eval/mnist_new_models_AdvIBP_04_eval.json
```
