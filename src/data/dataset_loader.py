import os
from datasets import load_dataset
from PIL import Image
import yaml
from pathlib import Path
import requests
import zipfile
import io
import numpy as np
import cv2

def adjust_bbox_coordinates(bbox, orig_width, orig_height, new_size, padding):
    """
    Adjust bounding box coordinates after image resizing and padding
    
    Args:
        bbox: Original bounding box [x, y, width, height] in normalized coordinates
        orig_width: Original image width
        orig_height: Original image height
        new_size: New image size after resizing (width, height)
        padding: Padding added (left, top)
    
    Returns:
        Adjusted bounding box coordinates [x, y, width, height] in normalized coordinates
    """
    new_width, new_height = new_size
    pad_left, pad_top = padding
    
    # Calculate scaling factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    # Convert normalized to absolute coordinates
    x = bbox[0] * orig_width
    y = bbox[1] * orig_height
    w = bbox[2] * orig_width
    h = bbox[3] * orig_height
    
    # Scale coordinates
    x = x * scale_x + pad_left
    y = y * scale_y + pad_top
    w = w * scale_x
    h = h * scale_y
    
    # Convert back to normalized coordinates (relative to target_size)
    target_size = max(new_width, new_height)
    x = x / target_size
    y = y / target_size
    w = w / target_size
    h = h / target_size
    
    return [x, y, w, h]

def resize_with_padding(image: Image.Image, target_size: int) -> tuple[Image.Image, tuple, tuple]:
    """
    Resize image to target size while maintaining aspect ratio and adding padding
    
    Returns:
        tuple: (resized image, new size (width, height), padding (left, top))
    """
    # Get original size
    orig_width, orig_height = image.size
    
    # Calculate scaling factor
    scale = min(target_size / orig_width, target_size / orig_height)
    
    # Calculate new size
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new square image with padding
    new_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))  # Gray padding
    
    # Calculate padding
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    
    # Paste resized image onto padded background
    new_image.paste(resized, (left, top))
    
    return new_image, (new_width, new_height), (left, top)

def get_dataset_info(dataset_name: str, split: str = "train") -> dict:
    """
    Get information about dataset size and structure
    """
    try:
        print(f"\nGetting info about {dataset_name} dataset...")
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        total_size = 0
        num_samples = len(dataset)
        
        # Check first item for size estimation
        if 'image' in dataset[0]:
            sample_image = dataset[0]['image']
            if isinstance(sample_image, Image.Image):
                img_byte_arr = io.BytesIO()
                sample_image.save(img_byte_arr, format='JPEG')
                sample_size = len(img_byte_arr.getvalue())
            else:
                sample_size = sample_image.nbytes
            
            total_size = sample_size * num_samples
            
        return {
            "num_samples": num_samples,
            "estimated_size_gb": total_size / (1024**3),
            "fields": dataset[0].keys()
        }
    except Exception as e:
        print(f"Error getting dataset info: {str(e)}")
        return None

def create_dataset_structure(base_dir: Path) -> tuple:
    """
    Create directory structure for dataset
    """
    train_dir = base_dir / "train" / "images"
    val_dir = base_dir / "valid" / "images"
    train_labels = base_dir / "train" / "labels"
    val_labels = base_dir / "valid" / "labels"
    
    for dir in [train_dir, val_dir, train_labels, val_labels]:
        dir.mkdir(parents=True, exist_ok=True)
        
    return train_dir, val_dir, train_labels, val_labels

def save_data_yaml(base_dir: Path, classes: dict):
    """
    Save data.yaml configuration file
    """
    data_yaml = {
        "path": str(base_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "names": classes
    }
    
    with open(base_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

def prepare_face_dataset(output_dir: str = "data/dataset", num_faces: int = 1000, image_size: int = 640):
    """
    Download and prepare face detection dataset
    
    Args:
        output_dir: Directory where to save the prepared dataset
        num_faces: Number of face images to use for training
        image_size: Size to resize images to (both width and height)
    """
    # Get dataset information
    faces_info = get_dataset_info("wider_face", f"train[:{num_faces}]")
    if not faces_info:
        return
        
    print(f"\nWIDER FACE dataset info:")
    print(f"Number of samples: {faces_info['num_samples']}")
    print(f"Estimated size: {faces_info['estimated_size_gb']:.2f} GB")
    print(f"Available fields: {faces_info['fields']}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with downloading the face dataset? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled by user")
        return

    # Create directory structure
    base_dir = Path(output_dir)
    train_dir, val_dir, train_labels, val_labels = create_dataset_structure(base_dir)
    
    try:
        # Download datasets
        print(f"\nDownloading WIDER FACE dataset (using {num_faces} images)...")
        faces_train = load_dataset("wider_face", split=f"train[:{num_faces}]", trust_remote_code=True)
        faces_val = load_dataset("wider_face", split="validation[:500]", trust_remote_code=True)
        
        # Process training set
        print("\nProcessing training images...")
        processed_train = process_wider_face_dataset(faces_train, train_dir, train_labels, image_size)
        
        # Process validation set
        print("\nProcessing validation images...")
        processed_val = process_wider_face_dataset(faces_val, val_dir, val_labels, image_size)
        
        print(f"\nFace dataset processing completed!")
        print(f"Processed {processed_train} training faces and {processed_val} validation faces")
        
        # Save configuration
        save_data_yaml(base_dir, {0: "face"})
        
        print("\nDataset preparation completed!")
        print(f"Dataset saved in: {base_dir.absolute()}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Full error details:", e.__class__.__name__)
        print("\nIf you're having issues with downloading datasets, you might want to:")
        print("1. Check your internet connection")
        print("2. Make sure you have access to HuggingFace datasets")
        print("3. Try running with a VPN if you're having access issues")
        print("4. Consider downloading datasets manually from their respective websites")

def prepare_logo_dataset(output_dir: str = "data/dataset", num_logos: int = 1000, image_size: int = 640):
    """
    Download and prepare logo detection dataset
    
    Args:
        output_dir: Directory where to save the prepared dataset
        num_logos: Number of logo images to use for training
        image_size: Size to resize images to (both width and height)
    """
    # Get dataset information
    logos_info = get_dataset_info("iamkaikai/amazing_logos_v4", f"train[:{num_logos}]")
    if not logos_info:
        return
        
    print(f"\nLogo dataset info:")
    print(f"Number of samples: {logos_info['num_samples']}")
    print(f"Estimated size: {logos_info['estimated_size_gb']:.2f} GB")
    print(f"Available fields: {logos_info['fields']}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with downloading the logo dataset? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled by user")
        return

    # Create directory structure
    base_dir = Path(output_dir)
    train_dir, val_dir, train_labels, val_labels = create_dataset_structure(base_dir)
    
    try:
        # Download datasets
        print(f"\nDownloading Logo dataset (using {num_logos} images)...")
        logos_train = load_dataset("iamkaikai/amazing_logos_v4", split=f"train[:{num_logos}]", trust_remote_code=True)
        logos_val = load_dataset("iamkaikai/amazing_logos_v4", split="train[-500:]", trust_remote_code=True)
        
        # Process training set
        print("\nProcessing training images...")
        processed_train = process_logo_dataset(logos_train, train_dir, train_labels, image_size)
        
        # Process validation set
        print("\nProcessing validation images...")
        processed_val = process_logo_dataset(logos_val, val_dir, val_labels, image_size)
        
        print(f"\nLogo dataset processing completed!")
        print(f"Processed {processed_train} training logos and {processed_val} validation logos")
        
        # Save configuration
        save_data_yaml(base_dir, {0: "logo"})
        
        print("\nDataset preparation completed!")
        print(f"Dataset saved in: {base_dir.absolute()}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Full error details:", e.__class__.__name__)
        print("\nIf you're having issues with downloading datasets, you might want to:")
        print("1. Check your internet connection")
        print("2. Make sure you have access to HuggingFace datasets")
        print("3. Try running with a VPN if you're having access issues")
        print("4. Consider downloading datasets manually from their respective websites")

def find_logo_box(img_np):
    """
    Find logo bounding box by detecting first non-white pixels from each side
    Returns coordinates in format: [x1, y1, x2, y2]
    """
    if len(img_np.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Threshold for white pixels (slightly below 255 to account for compression)
    white_thresh = 250
    height, width = gray.shape
    
    # Find first non-white pixel from left
    x1 = width
    for x in range(width):
        if np.any(gray[:, x] < white_thresh):
            x1 = x
            break
    
    # Find first non-white pixel from right
    x2 = 0
    for x in range(width-1, -1, -1):
        if np.any(gray[:, x] < white_thresh):
            x2 = x
            break
    
    # Find first non-white pixel from top
    y1 = height
    for y in range(height):
        if np.any(gray[y, :] < white_thresh):
            y1 = y
            break
    
    # Find first non-white pixel from bottom
    y2 = 0
    for y in range(height-1, -1, -1):
        if np.any(gray[y, :] < white_thresh):
            y2 = y
            break
    
    # Add small fixed padding (5 pixels)
    padding = 5
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width - 1, x2 + padding)
    y2 = min(height - 1, y2 + padding)
    
    # Check if we found valid boundaries
    if x1 >= x2 or y1 >= y2:
        return None
        
    return [x1, y1, x2, y2]

def apply_augmentation(image, bbox):
    """
    Apply augmentation to image and adjust bounding box
    Returns: (augmented_image, adjusted_bbox)
    """
    orig_width, orig_height = image.size
    img_np = np.array(image)
    
    # List of augmentations to apply
    augmentations = []
    
    # 1. Random brightness adjustment (30% chance)
    if np.random.random() < 0.3:
        value = np.random.uniform(0.7, 1.3)
        img_np = cv2.convertScaleAbs(img_np, alpha=value, beta=0)
        augmentations.append("brightness")
    
    # 2. Random rotation (-10 to 10 degrees, 20% chance)
    if np.random.random() < 0.2:
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((orig_width/2, orig_height/2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, M, (orig_width, orig_height), borderValue=(255,255,255))
        augmentations.append("rotation")
    
    # 3. Random scale (±10%, 30% chance)
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.9, 1.1)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize the image
        img_np = cv2.resize(img_np, (new_width, new_height))
        
        # Calculate padding sizes
        pad_height = max(0, orig_height - new_height)
        pad_width = max(0, orig_width - new_width)
        
        # Add padding only if needed
        if pad_height > 0 or pad_width > 0:
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            
            img_np = cv2.copyMakeBorder(img_np, 
                                      top_pad, bottom_pad,
                                      left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, 
                                      value=(255,255,255))
        
        # Adjust bbox
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * scale) + (pad_width // 2)
        y1 = int(y1 * scale) + (pad_height // 2)
        x2 = int(x2 * scale) + (pad_width // 2)
        y2 = int(y2 * scale) + (pad_height // 2)
        bbox = [x1, y1, x2, y2]
        augmentations.append("scale")
    
    return Image.fromarray(img_np), bbox

def process_logo_dataset(dataset, images_dir: Path, labels_dir: Path, image_size: int = 640) -> int:
    """
    Process Logo dataset images and annotations
    """
    processed = 0
    
    for idx, example in enumerate(dataset):
        try:
            if idx % 100 == 0:
                print(f"Processing image {idx}/{len(dataset)}")
            
            # Load image
            image = example['image']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Convert to numpy array for processing
            img_np = np.array(image)
            
            # Get original dimensions
            orig_width, orig_height = image.size
            
            # Find logo bounding box
            bbox = find_logo_box(img_np)
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                # Convert to normalized coordinates
                x1_norm = x1 / orig_width
                y1_norm = y1 / orig_height
                x2_norm = x2 / orig_width
                y2_norm = y2 / orig_height
                
                # Calculate center point and dimensions
                center_x = (x1_norm + x2_norm) / 2
                center_y = (y1_norm + y2_norm) / 2
                width = x2_norm - x1_norm
                height = y2_norm - y1_norm
                
                # Create bounding box in YOLO format [center_x, center_y, width, height]
                boxes = [[center_x, center_y, width, height]]
            else:
                # Fallback: use center with 90% size if no logo detected
                boxes = [[0.5, 0.5, 0.9, 0.9]]
            
            # Resize image with padding
            resized_image, new_size, padding = resize_with_padding(image, image_size)
            
            # Save resized image
            image_path = images_dir / f"logo_{idx}.jpg"
            resized_image.save(image_path)
            
            # Save annotations
            with open(labels_dir / f"logo_{idx}.txt", 'w') as f:
                for box in boxes:
                    try:
                        if len(box) == 4:
                            # Adjust box coordinates for final image size
                            adjusted_box = adjust_bbox_coordinates(box, orig_width, orig_height, new_size, padding)
                            # Ensure coordinates are within [0, 1] range
                            adjusted_box = [max(0, min(1, coord)) for coord in adjusted_box]
                            f.write(f"0 {' '.join(map(str, adjusted_box))}\n")
                            processed += 1
                    except Exception as box_error:
                        print(f"Warning: Error processing box in image {idx}: {box_error}")
                        continue
                        
        except Exception as img_error:
            print(f"Warning: Error processing image {idx}: {img_error}")
            continue
            
    return processed

def process_wider_face_dataset(dataset, images_dir: Path, labels_dir: Path, image_size: int = 640) -> int:
    """
    Process WIDER FACE dataset images and annotations
    
    Args:
        dataset: HuggingFace dataset
        images_dir: Directory to save images
        labels_dir: Directory to save labels
        image_size: Size to resize images to
    """
    processed = 0
    skipped_many_faces = 0
    skipped_small_faces = 0
    min_face_size = 0.05  # Minimalny rozmiar twarzy (5% szerokości/wysokości obrazu)
    
    for idx, example in enumerate(dataset):
        try:
            if idx % 100 == 0:
                print(f"Processing image {idx}/{len(dataset)}")
            
            # Process annotations - get boxes from 'faces' field
            faces = example.get('faces', {})
            if isinstance(faces, dict):
                boxes = faces.get('bbox', [])
            else:
                boxes = []
                for face in faces:
                    if isinstance(face, dict) and 'bbox' in face:
                        boxes.append(face['bbox'])
            
            # Skip images with more than 3 faces
            if len(boxes) > 3:
                skipped_many_faces += 1
                if skipped_many_faces % 100 == 0:
                    print(f"Skipped {skipped_many_faces} images with more than 3 faces")
                continue
            
            # Load image to get dimensions
            image = example['image']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            orig_width, orig_height = image.size
            
            # Check face sizes and filter out small faces
            valid_boxes = []
            has_small_face = False
            
            for box in boxes:
                if len(box) == 4:  # x, y, w, h
                    x, y, w, h = box
                    # Normalize dimensions
                    w_norm = w / orig_width
                    h_norm = h / orig_height
                    
                    # Skip if either width or height is too small
                    if w_norm < min_face_size or h_norm < min_face_size:
                        has_small_face = True
                        continue
                    
                    valid_boxes.append(box)
            
            # Skip image if it contains any small faces
            if has_small_face:
                skipped_small_faces += 1
                if skipped_small_faces % 100 == 0:
                    print(f"Skipped {skipped_small_faces} images with small faces")
                continue
            
            # Skip if no valid faces left
            if not valid_boxes:
                continue
            
            # Resize image
            resized_image, new_size, padding = resize_with_padding(image, image_size)
            
            # Save image
            image_path = images_dir / f"face_{processed:05d}.jpg"
            resized_image.save(image_path)
            
            labels = []
            for box in valid_boxes:
                # Convert from WIDER FACE format (x, y, w, h in pixels) to normalized YOLO format
                x, y, w, h = box
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, orig_width - w))
                y = max(0, min(y, orig_height - h))
                w = min(w, orig_width - x)
                h = min(h, orig_height - y)
                
                # Convert to normalized coordinates (0-1)
                x_norm = x / orig_width
                y_norm = y / orig_height
                w_norm = w / orig_width
                h_norm = h / orig_height
                
                # Convert to YOLO format (center_x, center_y, width, height)
                center_x = x_norm + (w_norm / 2)
                center_y = y_norm + (h_norm / 2)
                
                # Create normalized box
                normalized_box = [center_x, center_y, w_norm, h_norm]
                
                # Adjust box coordinates for padding and resizing
                adjusted_box = adjust_bbox_coordinates(normalized_box, orig_width, orig_height, new_size, padding)
                
                # Ensure final coordinates are within [0, 1] range
                adjusted_box = [max(0, min(1, coord)) for coord in adjusted_box]
                
                labels.append(f"0 {' '.join(map(str, adjusted_box))}")
            
            # Save labels
            label_path = labels_dir / f"face_{processed:05d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            continue
    
    print(f"\nTotal images processed: {processed}")
    print(f"Total images skipped (>5 faces): {skipped_many_faces}")
    print(f"Total images skipped (small faces): {skipped_small_faces}")
    return processed 