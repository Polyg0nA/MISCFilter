import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
# We do not import img_as_ubyte from skimage to minimize external dependencies.
# Instead, we perform the standard float to uint8 scaling manually.

# Import project utilities and model
import utils
from models.MISCFilterNet import MISCKernelNet as myNet
from models.layers import window_partitionx, window_reversex

# Custom dataset that handles image format conversion and reads images robustly
class CustomDeblurDataset(Dataset):
    def __init__(self, input_dir):
        super(CustomDeblurDataset, self).__init__()
        self.input_dir = input_dir
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.img_filenames = []
        
        if os.path.exists(input_dir):
            for x in sorted(os.listdir(input_dir)):
                if x.lower().endswith(valid_exts):
                    self.img_filenames.append(x)
                    
        self.size = len(self.img_filenames)
        if self.size == 0:
            print(f"[Warning] No valid images found in {input_dir}")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        filename = self.img_filenames[index]
        img_path = os.path.join(self.input_dir, filename)
        
        # Open image and convert to 3-channel RGB to handle grayscale or alpha channels
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img)
        
        name_without_ext = os.path.splitext(filename)[0]
        return img_tensor, name_without_ext

def main():
    parser = argparse.ArgumentParser(description='Memory-Efficient MISCFilter Deblurring')
    parser.add_argument('--input_dir', default='./test_images', type=str, help='Directory of blurred images')
    parser.add_argument('--output_dir', default='./results', type=str, help='Directory to save deblurred results')
    parser.add_argument('--weights', default='./checkpoints/GoPro.pth', type=str, help='Path to weights file (.pth)')
    parser.add_argument('--win_size', default=256, type=int, help='Window size (default 256)')
    parser.add_argument('--chunk_size', default=8, type=int, help='Chunk size for batch processing patches to prevent OOM')
    parser.add_argument('--inference_mode', default='normal', choices=['normal', 'eval'], 
                        help='normal: standard model; eval: folded model (load_checkpoint_compress_doconv)')
    parser.add_argument('--limit', default=-1, type=int, 
                        help='Limit the number of images to process (default -1: process all)')
    
    args = parser.parse_args()
    
    # 1. Setup GPU configuration
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible GPU detected! MISCFilter requires CUDA because of CuPy custom kernels.")
    
    device = torch.device('cuda')
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Initialize and Load Model
    print(f"Loading model with weights: {args.weights} (mode: {args.inference_mode})...")
    if args.inference_mode == 'eval':
        # Instantiates the model using eval layers (DOConv2d_eval)
        model = myNet(inference=True)
        utils.load_checkpoint_compress_doconv(model, args.weights)
    else:
        # Instantiates the standard model (inference=False)
        model = myNet(inference=False)
        utils.load_checkpoint(model, args.weights)
        
    model.to(device)
    model.eval()
    
    # 3. Setup Dataset and Dataloader
    dataset = CustomDeblurDataset(args.input_dir)
    if len(dataset) == 0:
        print(f"Error: No images found in {args.input_dir}. Please place images in the input directory.")
        return
        
    if args.limit > 0:
        print(f"Limiting processing to the first {args.limit} images.")
        dataset.img_filenames = dataset.img_filenames[:args.limit]
        dataset.size = len(dataset.img_filenames)
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    print(f"Found {len(dataset)} images to process. Starting restoration...")
    
    # 4. Deblurring Loop
    start_time = time.time()
    with torch.no_grad():
        for img_tensor, filename in tqdm(dataloader, desc="Restoring Images"):
            filename = filename[0]
            
            # Transfer tensor to CUDA
            img_tensor = img_tensor.to(device)
            _, _, Hx, Wx = img_tensor.shape
            
            # Step 4.1: Partition the large image into win_size x win_size patches
            # input_re shape: (num_patches, 3, win_size, win_size)
            input_re, batch_list = window_partitionx(img_tensor, args.win_size)
            num_patches = input_re.shape[0]
            
            # Step 4.2: Feed patches through the network in smaller chunks to avoid VRAM OOM
            restored_chunks = []
            for i in range(0, num_patches, args.chunk_size):
                # Clean CUDA cache regularly
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                
                chunk = input_re[i : min(i + args.chunk_size, num_patches)]
                
                # Forward pass
                outputs = model(chunk)
                
                # Handle output tuple/list for normal mode
                if isinstance(outputs, (list, tuple)):
                    # normal mode returns a tuple: (outputs[::-1], outputs_fil[::-1])
                    # outputs[0] is outputs[::-1], which is a list of tensors [out, out2, out3]
                    first_output = outputs[0]
                    if isinstance(first_output, (list, tuple)):
                        restored_chunk = first_output[0]  # scale 1 output (full resolution)
                    else:
                        restored_chunk = first_output
                else:
                    restored_chunk = outputs
                    
                restored_chunks.append(restored_chunk)
                
            # Concatenate all restored patches
            restored_patches = torch.cat(restored_chunks, dim=0)
            
            # Step 4.3: Reverse partition back to full image shape
            restored_img = window_reversex(restored_patches, args.win_size, Hx, Wx, batch_list)
            restored_img = torch.clamp(restored_img, 0.0, 1.0)
            
            # Convert to numpy array and save
            restored_img_np = restored_img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            restored_img_ubyte = (restored_img_np * 255.0).round().astype(np.uint8)
            
            # Save output image
            out_path = os.path.join(args.output_dir, f"{filename}_deblurred.png")
            utils.save_img(out_path, restored_img_ubyte)
            
    total_time = time.time() - start_time
    print(f"\nRestoration completed successfully in {total_time:.2f} seconds!")
    print(f"Average time per image: {total_time / len(dataset):.2f} seconds.")
    print(f"Restored images saved in: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
