# GANs
> Author： Hai Zhou\
> Institution： Beijing University of Posts and Telecommunication\
> mail：1583124882@qq.com 

Run experiments with various GAN model

## Model
We compare the following models:
* [x] [DCGAN](https://arxiv.org/abs/1511.06434)
* [x] [WGAN](https://arxiv.org/abs/1701.07875)
* [x] [SAGAN](https://arxiv.org/abs/1805.08318v2)
* [ ] [TransGAN](https://arxiv.org/abs/2102.07074)

### [DCGAN](https://arxiv.org/abs/1511.06434)

<p align="center"><img src="Image/DCGAN/dcgan.png" width="500" /></p>

Architecture guidelines for stable DCGAN
* Replace any pooling layers with strided convolutions(discriminator) and fractional-strided convolutions(generator).
* Use batchnorm in both the generator and the discriminator.
* Remove fully connected hihdden layers for deeper architectures.
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
* Use LeakyReLU activation in the discriminator for all layers.

### [WGAN](https://arxiv.org/abs/1701.07875)
Since GAN[^1] was born, it has been difficult to train. WGAN fundamentally solves this problem with four improvements.
* Remove the *Sigmoid* activation function of the discriminator.
* The loss function removes the *log* operation.
* After each update of the discriminator parameters truncate their absolute values to no more than a fixed constant c.
* Replace Adam optimizer with RMSProp.

### [SAGAN](https://arxiv.org/abs/1805.08318v2)
<p align="center"><img src="Image/SAGAN/sagan.png" width=500"></p>
The author proposes to use Self-Attention Generative Adversarial Network (SAGAN) for image generation tasks. In order to ensure the stability of model training, the author proposes

* apply spectral normalization to the GAN generator and as well as in the discriminator. 
* confirm that the twotimescale update rule (TTUR).

## Dataset
* A dataset of 30,000 face，[CelebA](https://drive.google.com/drive/folders/1YRRaC3LWLHorVhFNJPzVqLrUlA10eLEJ)

## Configuration Environment

`imageio==2.9.0`,\
`Pillow==9.2.0`, \
`torch==1.11.0`, \
`torchvision==0.12.0`

You can configure the environment with the command
```
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## Run
if your image dataset path is *../MyData/cv/CelebA/image/*
* DCGAN
```bash
torchrun --nproc_per_node=1 run.py \
--log_steps 10 \
--model dcgan \
--epochs 300 \
--batch_size 64 \
--data_path ../MyData/cv/CelebA/image/
```

* WGAN
```bash
torchrun --nproc_per_node=1 run.py \
--log_steps 10 \
--model wgan \
--epochs 300 \
--lr 5e-5 \
--batch_size 64 \
--data_path ../MyData/cv/CelebA/image/
```

* SAGAN
```bash
torchrun --nproc_per_node=3 run.py \
--log_steps 10 \
--model sagan \
--epochs 1000 \
--batch_size 64 \
--data_path ../MyData/cv/CelebA/image/
```

### How to test my model?
Before testing the model, please copy `gen.pt` and `dis.pt` in `./results` to the corresponding location in `./models` after training. Or you can directly download the model [here](https://drive.google.com/drive/folders/13OtGRV41ICQs4oxIsp-J9fxRfCoNsy_V?usp=sharing) and put it in the corresponding location of `./models`, such as `./models/CelebA/dcgan/gen.pt`.
And you can run

* DCGAN
```bash
torchrun --nproc_per_node=1 run.py --model dcgan --batch_size 64 --mode test
```

* WGAN
```bash
torchrun --nproc_per_node=1 run.py --model wgan --batch_size 64 --mode test
```

* SAGAN
```bash
torchrun --nproc_per_node=1 run.py --model sagan --batch_size 64 --mode test
```

## Experimental results
You can enjoy the results in `./results`.

* real example

<p align="center"><img src="Image/real.png" alt="真是样本" width="500" align=center /></p>

### DCGAN

<p align="center"><img src="Image/DCGAN/fake.png" width="500" align=center/></p>

### WGAN

<p align="center"><img src="Image/wgan/fake.png" width="500"/></p>

### SAGAN

<p align="center"><img src="Image/SAGAN/fake.png" width="500"></p>

### Evaluate
| Model |  Fid  | Time Cost |
|:-----:|:-----:|:---------:|
| DCGAN | 24.86 |   6h6m    |
| WGAN  | 18.55 |   5h56m   |
| SAGAN | 25.63 |   7h46m   |

**Notice:** *The above are the results of running 300 epochs in NVIDIA TITAN Xp (**1x**)*.

# References
* [WGAN Bolg](https://zhuanlan.zhihu.com/p/25071913) 
* [WGAN](https://arxiv.org/abs/1701.07875)
* [pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan)
* [SAGAN](https://arxiv.org/abs/1805.08318v2)