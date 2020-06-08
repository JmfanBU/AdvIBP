# Adversarial Training and Provable Robustness: A Tale of Two Objectives

*AdvIBP* is the official implementation of Adversarial Training and Provable Robustness: A Tale of Two Objectives. *AdvIBP* is a certifed adversarial training method that combines adversarial training and provable robustness verification. *AdvIBP* matches or outperforms state-of-art approach, [CROWN-IBP](https://openreview.net/pdf?id=Skxuk1rFwB), for provable L\_infinity robustness on MNIST and CIFAR.

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

### *AdvIBP* models:

| Dataset  | Test epsilon | Model path in trained_IBP_models          | Standard error | Verified error | PGD error |
|----------|--------------|-------------------------------------------|----------------|----------------|-----------|
| MNIST    | 0.1          | mnist/mnist_small_01_02<br>mnist/mnist_med_01_02<br>mnist/mnist_large_01_02| 1.63%<br>1.41%<br>1.03%        | 3.69%<br>3.24%<br>2.28%           | 2.70%<br>2.26%<br>1.53%                       |
| MNIST    | 0.2          | mnist/mnist_small_02_04<br>mnist/mnist_med_02_04<br>mnist/mnist_large_02_04| 4.15%<br>2.33%<br>1.58%        | 7.68%<br>5.37%<br>4.70%           | 5.81%<br>3.54%<br>2.59%                |
| MNIST    | 0.3          | mnist/mnist_small_03_04<br>mnist/mnist_med_03_04<br>mnist/mnist_large_03_04| 4.15%<br>2.33%<br>1.58%        | 10.80%<br>8.73%<br>8.23%           | 6.83%<br>4.35%<br>3.17%                |
| MNIST    | 0.4          | mnist/mnist_small_04_04<br>mnist/mnist_med_04_04<br>mnist/mnist_large_04_04| 4.15%<br>2.72%<br>1.88%        | 17.57%<br>16.18%<br>16.57%         | 8.48%<br>5.585%<br>3.23%               |
| CIFAR-10 | 2/255        | cifar-10/cifar_small_2_255<br>cifar-10/cifar_med_2_255<br>cifar-10/cifar_large_2_255| 42.33%<br>35.36%<br>40.61%        | 56.00%<br>52.27%<br>51.66%         | 50.08%<br>43.75%<br>46.97%               |
| CIFAR-10 | 8/255        | cifar-10/cifar_small_8_255<br>cifar-10/cifar_med_8_255<br>cifar-10/cifar_large_8_255| 57.88%<br>54.20%<br>52.86%        | 70.31%<br>68.21%<br>66.57%         | 66.52%<br>61.21%<br>61.66%               |
| CIFAR-10 | 16/255       | cifar-10/cifar_small_16_255<br>cifar-10/cifar_med_16_255<br>cifar-10/cifar_large_16_255| 67.32%<br>66.26%<br>64.40%        | 78.12%<br>77.79%<br>76.05%         | 73.44%<br>73.52%<br>71.78%               |
