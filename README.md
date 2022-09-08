# GANs
> Author： Hai Zhou\
> Institution： Beijing University of Posts and Telecommunication\
> mail：1583124882@qq.com 

Run experiments with various GAN model

## Model
### [DCGAN](https://arxiv.org/abs/1511.06434)

<img src="Image/DCGAN/dcgan.png" width="500" align=center/>

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

**notice⚠️️:** *The RMSProp optimizer is used in the original wgan, the batch_size is set to 64, and the training is 600,000 epochs. I have limited funds, set the batch_size to 500, use the Adam optimizer, and train 300epochs.*

## Dataset
* A dataset of 30,000 face，[CelebA](https://drive.google.com/drive/folders/1YRRaC3LWLHorVhFNJPzVqLrUlA10eLEJ)

## Configuration Environment
```
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## Run
if your dataset path is *./dataset/data/*
* DCGAN
```
torchrun --nproc_per_node=3 run.py --log_steps 10 --model dcgan --epochs 300 --data_path dataset/data/
```

* WGAN
```
torchrun --nproc_per_node=3 run.py --lr 5e-5 --log_steps 10 --model wgan --epochs 300 --data_path dataset/data/
```

## Experimental results
* real example

<img src="Image/real.png" alt="真是样本" width="500" align=center />

### DCGAN
* training process(10epoch/fps)

<img src="Image/DCGAN/train_epoch.gif" width="500" align=center />

* good result

<img src="Image/DCGAN/fake.png" width="500" align=center/>

### WGAN
* training process(10epoch/fps)

<img src="Image/wgan/train_epoch.gif" width="500" align=center />

* good result

<img src="Image/wgan/fake.png" width="500" align=center/>


**Notice:** *Continuing to train for more epochs should give better results, but is limited by money,
The training ends after 300 rounds.*

# References
* [WGAN Bolg](https://zhuanlan.zhihu.com/p/25071913) 
* [WGAN](https://arxiv.org/abs/1701.07875)
* [pytorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan)