import os
import torch.nn as nn
import imageio
import os
from tqdm import tqdm


def safe_create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compose_gif(path):
    files = [i for i in os.listdir(os.path.join(path, 'Img')) if 'fake' in i]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    gif_images = []
    for p in tqdm(files[::10], ncols=90):
        gif_images.append(imageio.imread(os.path.join(path, 'Img', p)))
    print("正在保存...")
    imageio.mimsave(os.path.join(path, 'train_epoch.gif'), gif_images, fps=3)
    print("保存成功...")


if __name__ == "__main__":
    DCGAN_img = 'results/CelebA/DCGAN/'
    wgan_img = 'results/CelebA/wgan/'
    compose_gif(wgan_img)
