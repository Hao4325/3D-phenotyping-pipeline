import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def process_images_to_masks_and_foreground(input_folder, output_mask_folder='masks', output_img_folder='img', threshold=128, channel='L'):
    # Ensure input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Create output folders
    Path(output_mask_folder).mkdir(parents=True, exist_ok=True)
    Path(output_img_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_formats):
            image_files.append(file)
    
    if not image_files:
        print(f"No supported image files found in folder {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    for i, filename in enumerate(image_files, 1):
        try:
            # Build full file path
            input_path = os.path.join(input_folder, filename)
            
            # Read image
            original_image = cv2.imread(input_path)
            if original_image is None:
                print(f"Unable to read image: {filename}")
                continue
            
            # Convert to CIELab color space
            lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
            
            # Split L, a, b channels
            L, a, b = cv2.split(lab_image)
            
            # Select channel for thresholding
            if channel == 'L':
                channel_data = L
            elif channel == 'a':
                channel_data = a
            elif channel == 'b':
                channel_data = b
            else:
                print(f"Unknown channel selection: {channel}, using L channel as default")
                channel_data = L
            
            # Apply threshold to generate binary mask
            _, binary_mask = cv2.threshold(channel_data, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations (optional, for noise removal)
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Extract foreground
            if len(original_image.shape) == 3:
                mask_3channel = cv2.merge([cleaned_mask, cleaned_mask, cleaned_mask])
                foreground = cv2.bitwise_and(original_image, mask_3channel)
            else:
                foreground = cv2.bitwise_and(original_image, cleaned_mask)
            
            # Create output filenames
            base_name = os.path.splitext(filename)[0]
            
            # Save mask image
            mask_filename = f"{base_name}.mask.png"
            mask_output_path = os.path.join(output_mask_folder, mask_filename)
            cv2.imwrite(mask_output_path, cleaned_mask)
            
            # Save foreground image
            img_filename = f"{base_name}.png"
            img_output_path = os.path.join(output_img_folder, img_filename)
            cv2.imwrite(img_output_path, foreground)
            
            # Optional: Save foreground image with transparent background (for color images)
            if len(original_image.shape) == 3:
                # Create image with alpha channel
                foreground_with_alpha = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
                # Use mask as alpha channel
                foreground_with_alpha[:, :, 3] = cleaned_mask
                
                # Save foreground image with alpha channel
                img_alpha_filename = f"{base_name}_foreground_alpha.png"
                img_alpha_output_path = os.path.join(output_img_folder, img_alpha_filename)
                # cv2.imwrite(img_alpha_output_path, foreground_with_alpha)
                
                print(f"Progress: {i}/{len(image_files)} - {filename} -> {mask_filename}, {img_filename}")
            else:
                print(f"Progress: {i}/{len(image_files)} - {filename} -> {mask_filename}, {img_filename}")
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Masks saved to: {output_mask_folder}")
    print(f"Foreground images saved to: {output_img_folder}")

def process_with_adaptive_threshold(input_folder, output_mask_folder='masks_adaptive', output_img_folder='img_adaptive', channel='L'):
    # Ensure input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Create output folders
    Path(output_mask_folder).mkdir(parents=True, exist_ok=True)
    Path(output_img_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_formats):
            image_files.append(file)
    
    if not image_files:
        print(f"No supported image files found in folder {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files, using adaptive thresholding")
    
    for i, filename in enumerate(image_files, 1):
        try:
            input_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(input_path)
            
            if original_image is None:
                print(f"Unable to read image: {filename}")
                continue
            
            # Convert to CIELab color space
            lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab_image)
            
            # Select channel
            if channel == 'L':
                channel_data = L
            elif channel == 'a':
                channel_data = a
            elif channel == 'b':
                channel_data = b
            else:
                channel_data = L
            
            # Apply adaptive thresholding
            binary_mask = cv2.adaptiveThreshold(
                channel_data, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Extract foreground
            if len(original_image.shape) == 3:
                mask_3channel = cv2.merge([binary_mask, binary_mask, binary_mask])
                foreground = cv2.bitwise_and(original_image, mask_3channel)
            else:
                foreground = cv2.bitwise_and(original_image, binary_mask)
            
            # Save results
            base_name = os.path.splitext(filename)[0]
            
            # Save mask
            mask_output_path = os.path.join(output_mask_folder, f"{base_name}_mask.png")
            cv2.imwrite(mask_output_path, binary_mask)
            
            # Save foreground
            img_output_path = os.path.join(output_img_folder, f"{base_name}_foreground.png")
            cv2.imwrite(img_output_path, foreground)
            
            print(f"Progress: {i}/{len(image_files)} - {filename}")
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
    
    print(f"\nAdaptive thresholding processing complete!")
    print(f"Masks saved to: {output_mask_folder}")
    print(f"Foreground images saved to: {output_img_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images to generate masks and foregrounds')
    parser.add_argument('--input_folder', type=str, default='./data/raw', help='Input folder path containing images')
    parser.add_argument('--output_mask_folder', type=str, default='./data/masks', help='Output folder for masks')
    parser.add_argument('--output_img_folder', type=str, default='./data/input', help='Output folder for foreground images')
    parser.add_argument('--threshold', type=int, default=30, help='Threshold value for binarization')
    parser.add_argument('--channel', type=str, default='L', choices=['L', 'a', 'b'], help='CIELab channel to use')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive thresholding instead of fixed threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Folder {args.input_folder} does not exist")
        exit(1)
    
    if args.adaptive:
        process_with_adaptive_threshold(
            input_folder=args.input_folder,
            output_mask_folder=args.output_mask_folder,
            output_img_folder=args.output_img_folder,
            channel=args.channel
        )
    else:
        process_images_to_masks_and_foreground(
            input_folder=args.input_folder,
            output_mask_folder=args.output_mask_folder,
            output_img_folder=args.output_img_folder,
            threshold=args.threshold,
            channel=args.channel
        )