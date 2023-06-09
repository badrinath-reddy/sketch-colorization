import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import os
import time
import datetime
import sys
from models.pix2pix import *
from data_loader import *
from loss import *
from torchvision.utils import save_image
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--model_name", type=str, default="pix2pix_simple", help="name of the model")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

writer = SummaryWriter()

os.makedirs("images/%s" % opt.model_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.model_name, exist_ok=True)

device = get_device()

# Loss functions
gen_loss = generator = MS_SSIM_L1_LOSS()
desc_loss = DiscriminatorLoss()


# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()


generator = generator.to(device)
discriminator = discriminator.to(device)
gen_loss = gen_loss.to(device)
desc_loss = desc_loss.to(device)


if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.model_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.model_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = get_data_loader(opt.batch_size, split='train')
val_dataloader = get_data_loader(opt.batch_size, split='val')

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    real_A, real_B = next(iter(val_dataloader))
    real_A = real_A.to(device)
    real_B = real_B.to(device)
    fake_B = generator(real_A)
    real_A = denormalize(real_A)
    real_B = denormalize(real_B)
    fake_B = denormalize(fake_B)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.model_name, batches_done), nrow=5, normalize=True)


# with torch.no_grad():
#     input, output = next(iter(dataloader))
#     input = input.to(device)
#     output = output.to(device)
#     writer.add_graph(generator, input)
#     writer.add_graph(discriminator, (input, output))


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)

        # Total loss
        loss_G = gen_loss(fake_B, real_B, pred_fake)

        loss_G.backward()
        
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)

        # Total loss
        loss_D = desc_loss(pred_real, pred_fake.detach())

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            writer.add_scalar('Loss/Generator', loss_G, batches_done)
            writer.add_scalar('Loss/Descriminator', loss_D, batches_done)
            writer.add_images('Images/Input', denormalize(real_A), batches_done)
            writer.add_images('Images/Output', denormalize(real_B), batches_done)
            writer.add_images('Images/Generator', denormalize(fake_B), batches_done)
            sample_images(batches_done)

            with torch.no_grad():
                real_A, real_B = next(iter(val_dataloader))
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                loss_G = gen_loss(fake_B, real_B, pred_fake)

                pred_real = discriminator(real_B, real_A)
                loss_D = desc_loss(pred_real, pred_fake)

            writer.add_scalar('Loss/Generator_val', loss_G, batches_done)
            writer.add_scalar('Loss/Descriminator_val', loss_D, batches_done)
            writer.add_images('Images/Generator_val', denormalize(fake_B), batches_done)
            writer.add_images('Images/Output_val', denormalize(real_B), batches_done)
            writer.add_images('Images/Input_val', denormalize(real_A), batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(),"saved_models/%s/generator_%d.pth" % (opt.model_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.model_name, epoch))

torch.save(generator.state_dict(),"saved_models/%s/generator_final.pth" % (opt.model_name))
torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_final.pth" % (opt.model_name))