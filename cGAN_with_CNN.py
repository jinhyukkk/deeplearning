from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

G_LR = 2e-4
D_LR = 5e-5
EPOCH = 200
BATCH_SIZE = 64

LATENT_DIM = 128


class Residual(nn.Module):
    def __init__(self, channel):
        super().__init__()
        """A simple residual block used by the networks."""
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.leaky = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        return self.leaky(out)


class Pooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        """Downsampling block with convolution and pooling."""
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.leaky = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        out_ = self.leaky(self.bn1(self.conv1(x)))
        return out_


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        """CNN discriminator conditioned on class labels."""
        # extra 10 channels for the condition label
        self.block_1 = nn.Sequential(Residual(11), Pooling(11, 8))
        self.block_2 = nn.Sequential(Residual(8), Pooling(8, 16))
        self.fc = nn.Linear(16, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.leaky = nn.LeakyReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y = F.one_hot(y, num_classes=10).float()
        y = y.view(y.size(0), 10, 1, 1)
        y = y.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, y), dim=1)

        out = self.block_1(x)
        out = self.block_2(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.sigmoid(out)


class Transpose(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        """Upsampling block using transposed convolution."""
        self.conv_trans = nn.ConvTranspose2d(in_channels=in_channel,
                                             out_channels=out_channel,
                                             kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.leaky = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        out_ = self.leaky(self.bn1(self.conv_trans(x)))
        out = self.bn2(self.conv1(out_))
        out = out + out_
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        """CNN generator conditioned on class labels."""
        # latent vector + 10 dimensional condition
        self.fc = nn.Linear(LATENT_DIM + 10, 7 * 7 * 16)
        self.bn = nn.BatchNorm1d(7 * 7 * 16)
        self.transpose_1 = Transpose(16, 8)
        self.transpose_2 = Transpose(8, 1)
        self.leaky = nn.LeakyReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        y = F.one_hot(y, num_classes=10).float()
        x = torch.cat((x, y), dim=1)
        out = self.leaky(self.bn(self.fc(x)))
        out = out.view(out.size(0), 16, 7, 7)
        out = self.leaky(self.transpose_1(out))
        out = self.transpose_2(out)
        return self.tanh(out)


def train_discriminator(disc, criterion, optimizer, real_data, real_labels, fake_data, fake_labels):
    """Update discriminator weights."""
    n = real_data.size(0)
    optimizer.zero_grad()
    pred_real = disc(real_data, real_labels)
    loss_real = criterion(pred_real, torch.ones(n, 1, device=real_data.device))
    loss_real.backward()
    pred_fake = disc(fake_data, fake_labels)
    loss_fake = criterion(pred_fake, torch.zeros(n, 1, device=real_data.device))
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake

def train_generator(disc, criterion, optimizer, fake_data, fake_labels):
    """Update generator weights."""
    n = fake_data.size(0)
    optimizer.zero_grad()
    pred = disc(fake_data, fake_labels)
    loss = criterion(pred, torch.ones(n, 1, device=fake_data.device))
    loss.backward()
    optimizer.step()
    return loss


def main():
    train_ds = MNIST('./', train=True, download=True,
                     transform=T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    disc.cuda()
    gen.cuda()
    criterion = nn.BCELoss().cuda()
    g_optim = torch.optim.Adam(gen.parameters(), lr=G_LR)
    d_optim = torch.optim.Adam(disc.parameters(), lr=D_LR)

    for epoch in range(EPOCH):
        gen.train()
        disc.train()
        for img, target in tqdm(train_loader, leave=False):
            img = img.cuda()
            target = target.cuda()
            n = img.size(0)
            noise = torch.randn(n, LATENT_DIM, device=img.device)
            fake = gen(noise, target).detach()
            disc_loss = train_discriminator(disc, criterion, d_optim, img, target, fake, target)

            noise = torch.randn(n, LATENT_DIM, device=img.device)
            fake = gen(noise, target)
            gen_loss = train_generator(disc, criterion, g_optim, fake, target)

        if not epoch % 10:
            print(disc_loss.item(), gen_loss.item())


if __name__ == '__main__':
    disc = Discriminator()
    gen = Generator()
    main()
