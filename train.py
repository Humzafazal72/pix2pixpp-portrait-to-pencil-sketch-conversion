import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from DatasetLoader import create_dataloader
from architecture import unetGenerator,Discriminator

# disc.load_state_dict(checkpoint['disc_state_dict'])
# gen.load_state_dict(checkpoint['gen_state_dict'])
# opt_disc.load_state_dict(checkpoint['discopt_state_dict'])
# opt_gen.load_state_dict(checkpoint['genopt_state_dict'])

def training_loop(EPOCHS, DEVICE, L1_LAMBDA, train_loader):
    for epoch in range(EPOCHS):
        running_loss_d = 0.0
        running_loss_g = 0.0
        for image,sketch in tqdm(train_loader,desc=f"EPOCH {epoch}:"):
            image,sketch = image.to(DEVICE),sketch.to(DEVICE)

            #Discriminator Training
            D_real = disc(image, sketch)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))

            fake = gen(image)
            D_fake = disc(image, fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))

            D_loss = (D_real_loss + D_fake_loss) / 2

            opt_disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            #Generator Training
            D_fake = disc(image, fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(fake, sketch) * L1_LAMBDA
            G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()  
            running_loss_d+=D_loss
            running_loss_g+=G_loss

        print('Generator Loss: ', running_loss_g/len(train_loader))
        print('Discriminator Loss: ', running_loss_d/len(train_loader))

if __name__=='__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir",help="provide the path to portraits",
                        default=0,type=int)
    parser.add_argument("tgt_dir",help="provide the path to sketches",
                        default='data\mock dataset\portrait',type=str)
    parser.add_argument("epochs",help="provide the number of epochs",
                        default='data\mock dataset\sketches',type=int)

    args = parser.parse_args()
    EPOCHS = args.epochs
    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)

    train_loader = create_dataloader(src_dir=src_dir,tgt_dir=tgt_dir)

    BCE = nn.BCEWithLogitsLoss()    
    L1_LOSS = nn.L1Loss()

    #checkpoint = torch.load('/kaggle/input/epoch-401/p2p_401.pt')
    disc = Discriminator(in_=6).to(DEVICE)
    gen = unetGenerator(in_=3, out_=64).to(DEVICE)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=100, betas=(0.5, 0.999))

    training_loop(EPOCHS = EPOCHS, DEVICE=DEVICE, L1_LAMBDA = 100, train_loader=train_loader)