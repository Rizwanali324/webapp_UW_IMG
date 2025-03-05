import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils

import time

from models import DenoisingDiffusionUWPhysical
from PIL import Image




def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--eta", type=float, default=0,
                        help="Number of implicit sampling steps")
    parser.add_argument('--seed', default=1234, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument("--condition_image", required=True, type=str,
                        help="Conditional Image")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def main():
    args, config = parse_args_and_config()
    to_tensor = torchvision.transforms.ToTensor()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    diffusion = DenoisingDiffusionUWPhysical(args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model_theta.eval()
    diffusion.model_phi.eval()

    with torch.no_grad():
      if os.path.isdir(args.condition_image):
          # Process directory of images
          x_cond_fnames = os.listdir(args.condition_image)
          for fname in x_cond_fnames:
              # Remove the folder path from the filename
              fname_ = os.path.splitext(fname)[0]
              fname = os.path.join(args.condition_image, fname)
              x_cond = Image.open(fname)
              x_cond = x_cond.resize((config.data.image_size, config.data.image_size), Image.Resampling.LANCZOS)
              x_cond = to_tensor(x_cond).to(diffusion.device)
              x_cond = data_transform(x_cond[None, :, :, :])
              x = torch.randn(x_cond.size(), device=diffusion.device)
              t = time.time()
              y_output, _x0, A, T, y0 = diffusion.sample_image_(x_cond, x, eta=args.eta)
              print(f"Total time taken: {time.time() - t}\n")
              y_output = inverse_data_transform(y_output)
              # Save the image with the result path (ignoring subfolder structure)
              utils.logging.save_image(y_output, f"results/{fname_}.png")
      else:
          # Handle a single image
          fname = args.condition_image
          fname_ = os.path.splitext(os.path.basename(fname))[0]  # Extract filename without path
          x_cond = Image.open(fname)
          x_cond = x_cond.resize((config.data.image_size, config.data.image_size), Image.Resampling.LANCZOS)
          x_cond = to_tensor(x_cond).to(diffusion.device)
          x_cond = data_transform(x_cond[None, :, :, :])
          x = torch.randn(x_cond.size(), device=diffusion.device)
          t = time.time()
          y_output, _x0, A, T, y0 = diffusion.sample_image_(x_cond, x, eta=args.eta)
          print(f"Total time taken: {time.time() - t}\n")
          y_output = inverse_data_transform(y_output)
          # Save the image with the result path (ignoring subfolder structure)
          utils.logging.save_image(y_output, f"results/{fname_}.png")

main()
