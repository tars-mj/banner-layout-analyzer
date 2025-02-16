import cv2
import numpy as np
from pathlib import Path
import argparse

def visualize_label(image_path: str, label_path: str, output_path: str = None):
    """
    Visualize label on the image using OpenCV
    
    Args:
        image_path: Path to the image
        label_path: Path to the label file
        output_path: Path to save visualization (optional)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"\nImage size: {width}x{height}")
    
    # Read labels
    try:
        with open(label_path, 'r') as f:
            labels = f.readlines()
            
        if not labels:
            print(f"Error: No labels found in {label_path}")
            return
            
        # Create copy for visualization
        img_with_boxes = img.copy()
        
        # Process each label
        for i, label_line in enumerate(labels):
            label = label_line.strip().split()
            if len(label) != 5:
                print(f"Error: Invalid label format in {label_path}, line {i+1}")
                continue
            
            # Parse normalized coordinates
            class_id = int(label[0])
            x_center = float(label[1])
            y_center = float(label[2])
            w = float(label[3])
            h = float(label[4])
            
            # Convert to pixel coordinates
            x_center_px = int(x_center * width)
            y_center_px = int(y_center * height)
            w_px = int(w * width)
            h_px = int(h * height)
            
            # Calculate box corners
            x1 = int(x_center_px - w_px/2)
            y1 = int(y_center_px - h_px/2)
            x2 = int(x_center_px + w_px/2)
            y2 = int(y_center_px + h_px/2)
            
            print(f"\nLabel {i+1} information:")
            print(f"Normalized coordinates: center=({x_center:.3f}, {y_center:.3f}), size=({w:.3f}, {h:.3f})")
            print(f"Pixel coordinates: center=({x_center_px}, {y_center_px}), size=({w_px}, {h_px})")
            print(f"Box corners: ({x1}, {y1}) -> ({x2}, {y2})")
            
            # Draw center point
            cv2.circle(img_with_boxes, (x_center_px, y_center_px), 3, (0, 0, 255), -1)
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text with coordinates
            text = f"Logo {i+1}"
            cv2.putText(img_with_boxes, text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Create comparison visualization
        comparison = np.hstack((img, img_with_boxes))
        
        # Show images
        cv2.imshow('Original vs Labeled', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, comparison)
            print(f"\nSaved visualization to: {output_path}")
            
    except Exception as e:
        print(f"Error processing {label_path}: {str(e)}")

def verify_dataset(dataset_dir: str, num_samples: int = 5):
    """
    Verify random samples from dataset
    
    Args:
        dataset_dir: Path to dataset directory
        num_samples: Number of random samples to verify
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / 'train' / 'images'
    labels_dir = dataset_dir / 'train' / 'labels'
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Create output directory
    output_dir = dataset_dir / 'label_verification'
    output_dir.mkdir(exist_ok=True)
    
    # Verify random samples
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_path in samples:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Warning: No label file for {img_path}")
            continue
            
        output_path = output_dir / f"verify_{img_path.stem}.png"
        print(f"\nVerifying {img_path.name}")
        visualize_label(img_path, label_path, str(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify dataset labels')
    parser.add_argument('--dataset', type=str, default='data/logos_dataset',
                      help='Path to dataset directory')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of random samples to verify')
    
    args = parser.parse_args()
    
    verify_dataset(args.dataset, args.samples) 