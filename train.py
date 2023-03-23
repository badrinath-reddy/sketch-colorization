import torch
from torch import nn
from models.simplegan import Generator, Discriminator
from utils import *
from data_loader import *
import os
from torch.utils.tensorboard import SummaryWriter

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1,
                    help="number of epochs of training")
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
parser.add_argument("--checkpoint_interval", type=int,
                    default=10, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

device = get_device()

writer = SummaryWriter()

ganloss = nn.MSELoss()
descloss = nn.L1Loss()


lambda_pixel = 100

generator = Generator().to(device)
descriminator = Discriminator().to(device)


dataloader = get_data_loader(opt.batch_size)

# Model to Tensorboard
input, _ = next(iter(dataloader))
input = input.to(device)

with torch.no_grad():
    writer.add_graph(generator, input)
    writer.add_graph(descriminator, (input, input))


writer.add_graph(generator, input)

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    descriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


step = 0
for i in range(opt.epoch):
    for j, (input, output) in enumerate(dataloader):
        step += 1

        input = input.to(device)
        output = output.to(device)

        valid_gt = torch.ones(input.size(0), 1, 16, 16).to(device)
        fake_gt = torch.zeros(input.size(0), 1, 16, 16).to(device)

        optimizer_G.zero_grad()

        generator_output = generator(input)
        fake_values = descriminator(input, generator_output)

        loss_G = ganloss(fake_values, valid_gt) + \
            lambda_pixel * descloss(fake_values, valid_gt)

        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_values = descriminator(input, output)

        loss_D = 0.5 * (descloss(real_values, valid_gt) +
                        descloss(fake_values.detach(), fake_gt))

        loss_D.backward()
        optimizer_D.step()

        writer.add_scalar('Loss/Generator', loss_G, step)
        writer.add_scalar('Loss/Descriminator', loss_D, step)

        if step % 100 == 0:
            writer.add_images('Images/Input', input, step)
            writer.add_images('Images/Output', output, step)
            writer.add_images('Images/Generator', generator_output, step)
