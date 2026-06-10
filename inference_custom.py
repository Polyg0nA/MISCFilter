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

# Helper functions for Geometric Self-Ensemble (Test-Time Augmentation)
def transform(x, i):
    # x: shape (B, C, H, W)
    if i == 0: return x
    elif i == 1: return torch.flip(x, [3]) # Flip Horizontal
    elif i == 2: return torch.flip(x, [2]) # Flip Vertical
    elif i == 3: return torch.rot90(x, 1, [2, 3]) # Rotate 90
    elif i == 4: return torch.rot90(torch.flip(x, [3]), 1, [2, 3]) # Flip H + Rotate 90
    elif i == 5: return torch.rot90(x, 2, [2, 3]) # Rotate 180
    elif i == 6: return torch.rot90(x, 3, [2, 3]) # Rotate 270
    elif i == 7: return torch.rot90(torch.flip(x, [3]), 3, [2, 3]) # Flip H + Rotate 270

def inv_transform(x, i):
    if i == 0: return x
    elif i == 1: return torch.flip(x, [3])
    elif i == 2: return torch.flip(x, [2])
    elif i == 3: return torch.rot90(x, -1, [2, 3]) # Rotate -90
    elif i == 4: return torch.flip(torch.rot90(x, -1, [2, 3]), [3])
    elif i == 5: return torch.rot90(x, -2, [2, 3])
    elif i == 6: return torch.rot90(x, -3, [2, 3])
    elif i == 7: return torch.flip(torch.rot90(x, -3, [2, 3]), [3])

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
        
        return img_tensor, filename

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
    parser.add_argument('--indices', default='', type=str, 
                        help='Comma-separated list of 0-based image indices to process (e.g., "0,2,4")')
    parser.add_argument('--filenames', default='', type=str, 
                        help='Comma-separated list of exact filenames to process (e.g., "img1.jpg,img2.png")')
    parser.add_argument('--self_ensemble', action='store_true', 
                        help='Use geometric self-ensemble (8 rotations/flips) to improve deblurring quality')
    
    args = parser.parse_args()
    
    # 1. Setup GPU configuration
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible GPU detected! MISCFilter requires CUDA because of CuPy custom kernels.")
    
    device = torch.device('cuda')
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    
    # Create directories (clear previous results to only keep current run outputs)
    if os.path.exists(args.output_dir):
        print(f"Cleaning previous results in output directory: {args.output_dir}...")
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Initialize and Load Model
    print(f"Loading model with weights: {args.weights} (mode: {args.inference_mode})...")
    if args.inference_mode == 'eval':
        model = myNet(inference=True)
        utils.load_checkpoint_compress_doconv(model, args.weights)
    else:
        model = myNet(inference=False)
        utils.load_checkpoint(model, args.weights)
        
    model.to(device)
    model.eval()
    
    # 3. Setup Dataset and Dataloader
    dataset = CustomDeblurDataset(args.input_dir)
    if len(dataset) == 0:
        print(f"Error: No images found in {args.input_dir}. Please place images in the input directory.")
        return
        
    # Filter by specific indices (0-based)
    if args.indices:
        try:
            target_indices = [int(idx.strip()) for idx in args.indices.split(',')]
            dataset.img_filenames = [dataset.img_filenames[i] for i in target_indices if 0 <= i < len(dataset.img_filenames)]
            dataset.size = len(dataset.img_filenames)
            print(f"Filtering dataset by specified indices: {target_indices}")
        except Exception as e:
            print(f"Error parsing --indices: {e}")
            
    # Filter by specific filenames
    if args.filenames:
        target_files = [f.strip() for f in args.filenames.split(',')]
        dataset.img_filenames = [f for f in dataset.img_filenames if f in target_files]
        dataset.size = len(dataset.img_filenames)
        print(f"Filtering dataset by specified filenames: {target_files}")
        
    # Slices dataset if limit is set (only if indices/filenames are not set)
    if args.limit > 0 and not args.indices and not args.filenames:
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
            
            # Setup Self-Ensemble
            restored_accum = torch.zeros_like(img_tensor)
            num_transforms = 8 if args.self_ensemble else 1
            
            for t_idx in range(num_transforms):
                # Apply transformation
                t_img = transform(img_tensor, t_idx)
                _, _, Hx, Wx = t_img.shape
                
                # Step 4.1: Partition the transformed image into win_size x win_size patches
                input_re, batch_list = window_partitionx(t_img, args.win_size)
                num_patches = input_re.shape[0]
                
                # Step 4.2: Feed patches through the network in smaller chunks
                restored_chunks = []
                for i in range(0, num_patches, args.chunk_size):
<<<<<<< HEAD
                    chunk = input_re[i : min(i + args.chunk_size, num_patches)]
                    
=======
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    
                    chunk = input_re[i : min(i + args.chunk_size, num_patches)]
                    
>>>>>>> cc0396741d51f05ba23f0b094f45fca28355d014
                    outputs = model(chunk)
                    
                    # Handle outputs list-nesting
                    if isinstance(outputs, (list, tuple)):
                        first_output = outputs[0]
                        if isinstance(first_output, (list, tuple)):
                            restored_chunk = first_output[0]  # scale 1 output (full resolution)
                        else:
                            restored_chunk = first_output
                    else:
                        restored_chunk = outputs
                        
                    restored_chunks.append(restored_chunk)
                    
                # Concatenate patches and reverse partition
                restored_patches = torch.cat(restored_chunks, dim=0)
                t_restored = window_reversex(restored_patches, args.win_size, Hx, Wx, batch_list)
                
                # Inverse transform back to original orientation
                restored_accum += inv_transform(t_restored, t_idx)
                
            # Average the ensemble results
            restored_img = restored_accum / num_transforms
            restored_img = torch.clamp(restored_img, 0.0, 1.0)
            
            # Convert to numpy array and save
            restored_img_np = restored_img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            restored_img_ubyte = (restored_img_np * 255.0).round().astype(np.uint8)
            
            # Save output image using the same file extension as the input to prevent size inflation
            name_part, ext_part = os.path.splitext(filename)
            out_filename = f"{name_part}_deblurred{ext_part}"
            out_path = os.path.join(args.output_dir, out_filename)
            utils.save_img(out_path, restored_img_ubyte)
            
            # Clean CUDA cache at the end of each image
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
    total_time = time.time() - start_time
    print(f"\nRestoration completed successfully in {total_time:.2f} seconds!")
    print(f"Average time per image: {total_time / len(dataset):.2f} seconds.")
    print(f"Restored images saved in: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
