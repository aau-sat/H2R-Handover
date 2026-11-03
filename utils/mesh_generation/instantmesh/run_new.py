
import os
import trimesh
import yaml
import numpy as np
import cv2
import torch
import time
import matplotlib.pyplot as plt

from PIL import Image

import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from utils.image_mesh_tools import preprocess_image, diffusion_image_generation, instant_mesh_process, align_mesh_to_coordinate

glctx = dr.RasterizeCudaContext()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('input_path', type=str, help='Path to input image or directory.')
    parser.add_argument('mask_input', type=str, help='Path to input mask image or directory.')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
    parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
    parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
    parser.add_argument('--rembg', action='store_true', default=True, help='Do not remove input background.')
    parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
    parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
    parser.add_argument('--obj_name', type=str, default='object', help='Name of the object.')
    args = parser.parse_args()
    seed_everything(args.seed)

    obj_name = args.obj_name

    if not os.path.isdir(args.input_path):
        raise ValueError(f'Input path {args.input_path} does not exist.')
    if not os.path.isdir(args.output_path):
        raise ValueError(f'Output path {args.output_path} is not a directory.')
    if not os.path.isdir(args.mask_input):
        raise ValueError(f'Mask input path {args.mask_input} is not a directory.')
    
    save_path = f'{args.output_path}/{args.obj_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # list files in the input path and take the first one
    input_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(input_files) == 0:
        raise ValueError(f'No image files found in {args.input_path}.')
    image = input_files[0]

    color = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    mask_files = [os.path.join(args.mask_input, f) for f in os.listdir(args.mask_input) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(mask_files) == 0:
        raise ValueError(f'No mask image files found in {args.mask_input}.')
    mask_image = mask_files[0]

    # mask = cv2.imread(os.path.join(demo_path, f"color_image_{obj_index}_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.bool_)
    mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE).astype(np.bool_)
    # print('Mask unique values:', np.unique(mask))
    # print('Mask shape:', mask.shape)
    # print('Mask dtype:', mask.dtype)
    # print(mask)
    # visualize color image, mask and depth image
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title('Color Image')
    plt.imshow(color)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    input_image = preprocess_image(color, mask, save_path, obj_name, rem_bg=args.rembg, flip=False)
    torch.cuda.empty_cache()
    images = diffusion_image_generation(save_path, save_path, obj_name, input_image=input_image, config_path=args.config)
    # time the process
    instant_mesh_process(images, save_path, obj_name, config_path=args.config)
    mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj_name}.obj'))
    mesh = align_mesh_to_coordinate(mesh)
    mesh.export(os.path.join(save_path, f'center_mesh_{obj_name}.obj'))

