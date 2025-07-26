#!/usr/bin/env python3.10

import sys
import os
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np

def view_tif_file(file_path, title=None):
    """View a TIF file using matplotlib"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    # Read the TIF file
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: Could not read {file_path}")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Min value: {img.min()}")
    print(f"Max value: {img.max()}")
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.colorbar(label='Pixel Value')
    
    if title:
        plt.title(title)
    else:
        plt.title(os.path.basename(file_path))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def list_available_tifs():
    """List all available TIF files in the data directory"""
    print("Available TIF files:")
    print("\nTerrain data (in each fire directory):")
    terrain_files = ['dem.tif', 'aspect.tif', 'slope.tif', 'ndvi.tif', 
                     'band_2.tif', 'band_3.tif', 'band_4.tif', 'band_5.tif']
    for file in terrain_files:
        print(f"  {file}")
    
    print("\nFire perimeter data (in perims/ subdirectory):")
    print("  Date-based files (e.g., 0711.tif, 0712.tif, etc.)")
    
    # List fires
    fires = [f for f in os.listdir('data') if os.path.isdir(os.path.join('data', f)) and not f.startswith('.')]
    print(f"\nAvailable fires: {fires}")

def main():
    parser = argparse.ArgumentParser(description='View TIF files in the FireCast project')
    parser.add_argument('--file', type=str, help='Path to TIF file to view')
    parser.add_argument('--fire', type=str, help='Fire name (e.g., beaverCreek)')
    parser.add_argument('--type', type=str, choices=['dem', 'aspect', 'slope', 'ndvi', 'band_2', 'band_3', 'band_4', 'band_5'], 
                       help='Type of terrain data to view')
    parser.add_argument('--date', type=str, help='Date for perimeter data (e.g., 0711)')
    parser.add_argument('--list', action='store_true', help='List available TIF files')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tifs()
        return
    
    if args.file:
        # Direct file path provided
        view_tif_file(args.file)
    elif args.fire and args.type:
        # View terrain data for specific fire
        file_path = f"data/{args.fire}/{args.type}.tif"
        view_tif_file(file_path, f"{args.fire} - {args.type}")
    elif args.fire and args.date:
        # View perimeter data for specific fire and date
        file_path = f"data/{args.fire}/perims/{args.date}.tif"
        view_tif_file(file_path, f"{args.fire} - Perimeter {args.date}")
    else:
        print("Usage examples:")
        print("  python3.10 view_tif.py --list")
        print("  python3.10 view_tif.py --fire beaverCreek --type dem")
        print("  python3.10 view_tif.py --fire beaverCreek --date 0711")
        print("  python3.10 view_tif.py --file data/beaverCreek/dem.tif")

if __name__ == "__main__":
    main() 