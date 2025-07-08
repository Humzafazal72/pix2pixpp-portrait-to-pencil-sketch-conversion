# How to take inference and save sketch
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from architecture import unetGenerator,unetppGenerator

def get_inference(img_path:Path,gen:torch.nn.Module):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1))/255.0
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img,dtype=torch.float32)

    # use your trained Generator or our trained weights.

    gen.eval()
    with torch.no_grad():
        pred = gen(img)
    
    pred = np.array(img.detach().to('cpu'))
    pred = pred[0].transpose(1, 2, 0)

    pred = np.clip(pred, 0.0, 1.0) #clip the negative pixels 
    plt.imsave(r'generated.jpg', pred)

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument("gen_type",help="Provide the generator type",
                        options=['unet','unet++'],type=str)
    parser.add_argument("gen_weights_path",help="Provide the path to the generator weights",
                        type=str)
    parser.add_argument("img_path",help="Provide the path to the input image",
                        type=str)
    
    args = parser.parse_args()
    gen_type = argparse.gen_type
    img_path = Path(args.img_path)
    gen_weights_path = Path(args.gen_weights_path)
    
    if img_path.exists() == False:
        print(f"The path {img_path} does not exist")
        exit()

    if gen_weights_path.exists() == False:
        print(f"The path {gen_weights_path} does not exist")
        exit()

    checkpoint = torch.load(gen_weights_path,map_location=DEVICE)
    if gen_type == 'unet':
        gen = unetGenerator.to(device=DEVICE)
    else:
        gen = unetppGenerator.to(device = DEVICE)

    gen.load_state_dict(checkpoint)
    
    get_inference(img_path=img_path, gen=gen)
