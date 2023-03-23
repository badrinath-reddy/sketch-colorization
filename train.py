import torch
from torch import nn
from models.simplegan import Generator, Discriminator
from utils import *
from data_loader import *
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--dataset_name", type=str,
                    default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256,
                    help="size of image height")
parser.add_argument("--img_width", type=int, default=256,
                    help="size of image width")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int,
                    default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

device = get_device()


ganloss = nn.MSELoss()
descloss = nn.L1Loss()

lambda_pixel = 100

generator = Generator().to(device)
descriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    descriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


dataloader = get_data_loader(opt.batch_size)

input, output = next(iter(dataloader))

input = input.to(device)
output = output.to(device)

fake = generator(input)

print(fake.shape)

vals = descriminator(input, output)
vals2 = descriminator(input, fake)

print(vals.shape, vals2.shape)
