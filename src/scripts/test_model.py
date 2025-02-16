import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
import torch

def tensor_to_float(value):
    """Convert tensor to float if needed"""
    if isinstance(value, torch.Tensor):
        return float(value.cpu().numpy())
    return float(value)

def validate_coordinates(x1, y1, x2, y2, img_shape):
    """
    Validate if coordinates are within image bounds and convert tensors to float
    """
    height, width = img_shape[:2]
    
    # Convert tensors to float
    x1, y1, x2, y2 = map(tensor_to_float, [x1, y1, x2, y2])
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    
    return {
        "original": {
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2
        },
        "normalized": {
            "x1": x1 / width,
            "y1": y1 / height,
            "x2": x2 / width,
            "y2": y2 / height
        },
        "is_valid": (x2 > x1 and y2 > y1)  # Additional check for valid box dimensions
    }

def test_model(model_path: str, image_paths: list, output_dir: str):
    """
    Test model on sample images
    
    Args:
        model_path: Path to the trained model weights
        image_paths: List of paths to test images
        output_dir: Directory to save visualization results
    """
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Set detection parameters
    model.conf = 0.7     # High confidence threshold for reliable detections
    model.iou = 0.5     # Increased IoU threshold for better separation
    model.max_det = 10  # Increased max detections
    model.verbose = True
    
    # Set class-specific confidence thresholds
    model.classes = [0, 1]  # Enable detection of both faces and logos
    model.agnostic = True   # Class-agnostic NMS
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in image_paths:
        try:
            img_path = str(Path(img_path).absolute())
            print(f"\nProcessing image: {img_path}")
            
            # Check if file exists
            if not Path(img_path).exists():
                print(f"Error: Image file not found: {img_path}")
                continue
            
            print(f"File exists, size: {Path(img_path).stat().st_size} bytes")
                
            # Load image for visualization
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image: {img_path}")
                print("Make sure the file is a valid image file")
                continue
            
            print(f"Image loaded successfully. Shape: {img.shape}")
            
            # Run inference
            results = model(img)
            
            print("\nDetection Results:")
            print("------------------")
            
            # Define class names
            class_names = {0: 'face', 1: 'logo'}
            
            # Prepare detections for JSON
            detections = []
            for i, box in enumerate(results[0].boxes.data):
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                class_name = class_names.get(class_id, f'unknown_{class_id}')
                
                # Validate coordinates
                coord_info = validate_coordinates(x1, y1, x2, y2, img.shape)
                
                print(f"\nDetection {i+1}:")
                print(f"Class: {class_name}")
                print(f"Original coordinates: ({coord_info['original']['x1']:.1f}, {coord_info['original']['y1']:.1f}) -> ({coord_info['original']['x2']:.1f}, {coord_info['original']['y2']:.1f})")
                print(f"Normalized coordinates: ({coord_info['normalized']['x1']:.3f}, {coord_info['normalized']['y1']:.3f}) -> ({coord_info['normalized']['x2']:.3f}, {coord_info['normalized']['y2']:.3f})")
                print(f"Confidence: {tensor_to_float(conf):.3f}")
                print(f"Valid coordinates: {coord_info['is_valid']}")
                
                detection_data = {
                    "coordinates": coord_info['original'],
                    "normalized_coordinates": coord_info['normalized'],
                    "confidence": tensor_to_float(conf),
                    "class": class_id,
                    "class_name": class_name,
                    "is_valid": coord_info['is_valid']
                }
                detections.append(detection_data)
            
            # Save detections to JSON
            json_path = output_dir / f'detection_{Path(img_path).stem}.json'
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
            
            print(f"\nSaved detection data to: {json_path}")
            
            # Convert color for visualization
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Plot results
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            
            # Draw detections
            num_detections = 0
            colors = {'face': 'red', 'logo': 'blue', 'unknown': 'gray'}
            for detection in detections:
                if detection['is_valid']:
                    coords = detection['coordinates']
                    x1, y1 = coords['x1'], coords['y1']
                    width = coords['x2'] - coords['x1']
                    height = coords['y2'] - coords['y1']
                    
                    class_name = detection['class_name']
                    color = colors.get(class_name, colors['unknown'])
                    
                    plt.gca().add_patch(plt.Rectangle(
                        (x1, y1), width, height,
                        fill=False, color=color, linewidth=2
                    ))
                    plt.text(
                        x1, y1-5, 
                        f'{class_name} {detection["confidence"]:.2f}',
                        color=color, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                    num_detections += 1
                else:
                    print(f"Warning: Skipping invalid detection with coordinates: {detection['coordinates']}")
            
            plt.title(f'Detections for {Path(img_path).name}\nFound {num_detections} valid detections')
            plt.axis('off')
            
            # Save result
            output_path = output_dir / f'detection_{Path(img_path).stem}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Processed successfully. Found {num_detections} valid detections")
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            print("Full error details:", e.__class__.__name__)
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained model on sample images')
    parser.add_argument('--model', type=str, default='runs/detect/combined_detector/weights/best.pt',
                      help='Path to model weights')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                      help='Paths to test images')
    parser.add_argument('--output', type=str, default='test_results',
                      help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        image_paths=args.images,
        output_dir=args.output
    ) 