import os
import torch.nn as nn
import imageio
import os
from tqdm import tqdm
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from dataset.dataLoader import DL
from torchvision.utils import save_image


def normalize_gradient(Dis, x, *arg, **args):
    b = x.shape[0]
    x.requires_grad_(True)
    f = Dis(x, *arg, **args)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True,
    )[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def safe_create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
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


def create_evaluate_npz(args):
    data = DL(args)
    dl = data.dl
    dataset = None
    if 'CelebA' in args.data_path:
        dataset = 'CelebA'
    num = 0
    safe_create_dir('results/%s/%s/' % (dataset, 'val_img'))
    print('extract img...')
    for batch, inputs in enumerate(dl):

        for img in tqdm(inputs, ncols=100):
            save_image((img*0.5)+0.5, 'results/%s/%s/%s.png' % (dataset, 'val_img', num))
            num += 1
            if num >= 3000:
                return


if __name__ == "__main__":
    DCGAN_img = 'results/CelebA/dcgan/Img/'
    wgan_img = 'results/CelebA/wgan/Img/'
    compose_gif(DCGAN_img)
