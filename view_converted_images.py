#!/usr/bin/env python3.10

import numpy as np
import matplotlib.pyplot as plt
import cv2

def view_converted_image(npy_path, png_path=None):
    """View a converted image and compare with original if available"""
    print(f"Loading: {npy_path}")
    
    # Load the NPY file
    img = np.load(npy_path)
    print(f"Shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Value range: {img.min()} to {img.max()}")
    print(f"Unique values: {np.unique(img)}")
    print(f"Fire pixels (1s): {np.sum(img)}")
    print(f"Background pixels (0s): {np.sum(img == 0)}")
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f"Converted Image: {npy_path}")
    plt.colorbar()
    plt.axis('off')
    
    # Save the display
    output_path = npy_path.replace('.npy', '_display.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved display image: {output_path}")
    plt.show()
    
    return img

def main():
    # View both converted images
    converted_images = [
        "madre/perims_converted/0707.npy",
        "madre/perims_converted/0708.npy"
    ]
    
    for npy_path in converted_images:
        print(f"\n{'='*60}")
        img = view_converted_image(npy_path)
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main() 