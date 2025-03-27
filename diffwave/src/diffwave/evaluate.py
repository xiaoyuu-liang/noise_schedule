from argparse import ArgumentParser
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from scipy.linalg import norm
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def compute_ssim(spectrogram1, spectrogram2):
    
    return ssim(spectrogram1, spectrogram2, data_range=spectrogram1.max() - spectrogram1.min())


def compute_fid(model, real_images, fake_images):
    real_features = model(real_images.repeat(1, 3, 1, 1)).detach().cpu().numpy()
    fake_features = model(fake_images.repeat(1, 3, 1, 1)).detach().cpu().numpy()
    print(real_features.shape, fake_features.shape)
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean = np.sqrt(sigma_real @ sigma_fake)
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2*covmean)

    return fid

def load_inception_model():
    model = models.inception_v3(pretrained=True, transform_input=False).eval()
    return model

def load_npy_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav.spec.npy')]
    files.sort() 
    return [os.path.join(folder_path, f) for f in files]

def pair_files(folder_a, folder_b):
    files_a = load_npy_files(folder_a)
    files_b = load_npy_files(folder_b)
    
    paired_files = []
    for file_a in files_a:
        filename = os.path.basename(file_a)
        corresponding_file = os.path.join(folder_b, filename.replace('.wav.spec.npy', '.out.wav.spec.npy'))
        if corresponding_file in files_b:
            paired_files.append((file_a, corresponding_file))
    
    return paired_files


def main(args):
    folder_a = args.data
    folder_b = args.output
    
    model = load_inception_model()
    total_ssim = 0
    total_fid = 0
    num_pairs = 0
    
    paired_files = pair_files(folder_a, folder_b)

    for file_a, file_b in paired_files:
        array_a = np.load(file_a)
        array_b = np.load(file_b)[:, :-1]

        if array_a.ndim == 2 and array_b.ndim == 2:
            # print(array_a.shape, array_b.shape)
            ssim_value = compute_ssim(array_a, array_b)
            total_ssim += ssim_value
        
            # real_tensor = torch.tensor(array_a).unsqueeze(0).float()
            # fake_tensor = torch.tensor(array_b).unsqueeze(0).float()
            
            # fid_value = compute_fid(model, real_tensor, fake_tensor)
            # total_fid += fid_value
            
            num_pairs += 1

    avg_ssim = total_ssim / num_pairs if num_pairs > 0 else 0
    avg_fid = total_fid / num_pairs if num_pairs > 0 else 0
    
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average FID: {avg_fid}")

if __name__ == '__main__':
  parser = ArgumentParser(description='evaluate SSIM/FID of given Spectrogram')
  parser.add_argument('--data', '-d', default='data',
        help='directory containing .wav.spec.npy files')
  parser.add_argument('--output', '-o', default='output',
        help='directory containing .out.wav.spec.npy files')
  main(parser.parse_args())