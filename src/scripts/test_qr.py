import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import json

def test_qr_detection(image_path: str, output_dir: str = "test_results"):
    """
    Test QR code detection on a single image
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"\nProcessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Initialize QR detector
    qr_detector = cv2.QRCodeDetector()
    
    # Detect QR codes
    print("\nDetecting QR codes...")
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(image)
    
    if not retval:
        print("No QR codes detected")
        return
        
    # Process detections
    detections = []
    for qr_points, qr_data in zip(points, decoded_info):
        if qr_points is not None:
            # Get bounding box coordinates
            x_coords = qr_points[:, 0]
            y_coords = qr_points[:, 1]
            x1, y1 = float(min(x_coords)), float(min(y_coords))
            x2, y2 = float(max(x_coords)), float(max(y_coords))
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "data": qr_data if qr_data else "QR Code detected",
                "points": qr_points.tolist()
            }
            detections.append(detection)
            
            print(f"\nDetected QR code:")
            print(f"Data: {detection['data']}")
            print(f"Bounding box: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
    
    # Save detections to JSON
    json_path = output_dir / f'qr_detection_{Path(image_path).stem}.json'
    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"\nSaved detection data to: {json_path}")
    
    # Visualize results
    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Draw detections
    for detection in detections:
        # Draw bounding box
        x1, y1, x2, y2 = detection["bbox"]
        width = x2 - x1
        height = y2 - y1
        
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), width, height,
            fill=False, color='blue', linewidth=2
        ))
        
        # Draw QR code corners
        points = np.array(detection["points"])
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
        plt.plot([points[-1, 0], points[0, 0]], 
                [points[-1, 1], points[0, 1]], 'r-', linewidth=2)
        
        # Add text with decoded data
        plt.text(
            x1, y1-10, 
            f'QR: {detection["data"]}',
            color='blue',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.title(f'QR Code Detection Results\nFound {len(detections)} QR codes')
    plt.axis('off')
    
    # Save visualization
    output_path = output_dir / f'qr_detection_{Path(image_path).stem}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test QR code detection')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image file')
    parser.add_argument('--output', type=str, default='test_results',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    test_qr_detection(args.image, args.output)

if __name__ == "__main__":
    main() 