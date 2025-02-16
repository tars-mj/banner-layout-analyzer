import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import colorsys

def find_tight_bbox(image):
    """
    Find tight bounding box coordinates for non-white pixels in the image
    
    Args:
        image: PIL Image in RGBA mode
    Returns:
        tuple (x1, y1, x2, y2) of tight bounding box coordinates
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # For RGBA images, consider pixel non-white if it's not fully transparent
    # and not completely white in RGB channels
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        # Pixel is considered part of logo if it's not transparent and not white
        mask = (alpha > 0) & ~np.all(rgb == [255, 255, 255], axis=2)
    else:  # RGB
        # Pixel is considered part of logo if it's not white
        mask = ~np.all(img_array == [255, 255, 255], axis=2)
    
    # Find non-white pixel coordinates
    y_indices, x_indices = np.where(mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    
    # Get bounding box coordinates
    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    
    return (x1, y1, x2, y2)

def check_overlap(new_box, existing_boxes, margin=20):
    """
    Check if new bounding box overlaps with existing ones
    Args:
        new_box: (x1, y1, x2, y2) of new box
        existing_boxes: list of (x1, y1, x2, y2) of existing boxes
        margin: minimum pixels between boxes
    """
    new_x1, new_y1, new_x2, new_y2 = new_box
    
    # Add margin to new box
    new_x1 -= margin
    new_y1 -= margin
    new_x2 += margin
    new_y2 += margin
    
    for box in existing_boxes:
        x1, y1, x2, y2 = box
        
        # Check if boxes overlap
        if not (new_x2 < x1 or new_x1 > x2 or new_y2 < y1 or new_y1 > y2):
            return True
            
    return False

def generate_gradient_background(size, start_color=None, end_color=None, direction='horizontal'):
    """
    Generate a gradient background
    
    Args:
        size: Tuple of (width, height)
        start_color: RGB tuple for start color (if None, generates soft random color)
        end_color: RGB tuple for end color (if None, generates soft random color)
        direction: 'horizontal', 'vertical', or 'diagonal'
    """
    if start_color is None:
        # Generate soft, pastel-like colors
        h = random.random()  # Random hue
        s = random.uniform(0.2, 0.4)  # Low saturation for softness
        v = random.uniform(0.9, 1.0)  # High value for brightness
        rgb = colorsys.hsv_to_rgb(h, s, v)
        start_color = tuple(int(x * 255) for x in rgb)
    
    if end_color is None:
        # Generate matching end color
        h = (h + random.uniform(0.1, 0.2)) % 1.0  # Slight hue shift
        s = random.uniform(0.2, 0.4)
        v = random.uniform(0.9, 1.0)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        end_color = tuple(int(x * 255) for x in rgb)
    
    # Create base image
    image = Image.new('RGB', size)
    draw = ImageDraw.Draw(image)
    
    # Generate gradient
    for i in range(size[0] if direction == 'horizontal' else size[1]):
        progress = i / (size[0] if direction == 'horizontal' else size[1])
        color = tuple(int(start + (end - start) * progress)
                     for start, end in zip(start_color, end_color))
        
        if direction == 'horizontal':
            draw.line([(i, 0), (i, size[1])], fill=color)
        elif direction == 'vertical':
            draw.line([(0, i), (size[0], i)], fill=color)
        else:  # diagonal
            draw.line([(i, 0), (i, size[1])], fill=color)
    
    return image

def generate_noise_background(size):
    """
    Generate a subtle noise texture background with random scale and intensity
    """
    # Create base white image
    image = Image.new('RGB', size, 'white')
    pixels = image.load()
    
    # Losowa skala szumu (wielkość ziarna)
    scale = random.uniform(10, 50)  # Od drobnego do grubego ziarna
    
    # Losowa intensywność szumu
    intensity = random.uniform(0.05, 0.15)  # Od subtelnego do bardziej widocznego
    
    # Generuj bazowy kolor tła (delikatnie off-white)
    base_color = random.randint(250, 255)
    
    # Add noise with random parameters
    for x in range(size[0]):
        for y in range(size[1]):
            # Używamy scale do kontroli "ziarnistości" szumu
            noise_value = random.uniform(-scale, scale) * intensity
            
            # Aplikuj szum do bazowego koloru
            pixel_value = int(max(0, min(255, base_color + noise_value)))
            pixels[x, y] = (pixel_value, pixel_value, pixel_value)
    
    return image

def generate_clouds_background(size, num_clouds=10):
    """
    Generate a background with soft cloud-like shapes
    """
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    for _ in range(num_clouds):
        # Random cloud parameters
        x = random.randint(-50, size[0])
        y = random.randint(-50, size[1])
        radius = random.randint(50, 150)
        opacity = random.randint(10, 30)  # Very subtle clouds
        
        # Create cloud shape using multiple overlapping circles
        for _ in range(5):
            offset_x = random.randint(-20, 20)
            offset_y = random.randint(-20, 20)
            cloud_radius = radius + random.randint(-20, 20)
            
            # Create subtle grey color for cloud
            grey = random.randint(245, 255)
            color = (grey, grey, grey, opacity)
            
            draw.ellipse([x + offset_x, y + offset_y, 
                         x + offset_x + cloud_radius, 
                         y + offset_y + cloud_radius],
                        fill=color)
    
    return image

def generate_solid_background(size):
    """
    Generate a solid background with soft, pastel-like colors
    
    Args:
        size: Tuple of (width, height)
    Returns:
        PIL Image with solid pastel color
    """
    # Generate soft, pastel-like color using HSV
    h = random.random()  # Random hue
    s = random.uniform(0.1, 0.3)  # Low saturation for pastel effect
    v = random.uniform(0.95, 1.0)  # High value for brightness
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(h, s, v)
    
    # Convert to 0-255 range
    color = tuple(int(x * 255) for x in rgb)
    
    return Image.new('RGB', size, color)

def generate_blurred_noise_background(size):
    """
    Generate background with large-scale blurred noise
    """
    # Losowe parametry dla dużego ziarna
    grain_scale = random.uniform(80, 150)  # Duże ziarno
    intensity = random.uniform(0.1, 0.2)  # Intensywność szumu
    
    # Losowy kolor bazowy (bardzo jasny)
    base_h = random.random()  # Losowy odcień
    base_s = random.uniform(0.05, 0.15)  # Małe nasycenie
    base_v = random.uniform(0.95, 1.0)  # Wysoka jasność
    base_rgb = colorsys.hsv_to_rgb(base_h, base_s, base_v)
    base_color = tuple(int(x * 255) for x in base_rgb)
    
    # Generuj szum jako obraz RGB
    noise_array = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
    
    # Dodaj losowy szum
    for x in range(size[0]):
        for y in range(size[1]):
            noise_value = random.uniform(-grain_scale, grain_scale) * intensity
            # Aplikuj szum do każdego kanału
            noise_array[y, x] = [
                int(max(0, min(255, c + noise_value))) for c in base_color
            ]
    
    # Konwertuj na obraz PIL i zastosuj rozmycie
    noise_image = Image.fromarray(noise_array, 'RGB')
    
    # Losowy promień rozmycia
    blur_radius = random.uniform(30, 70)
    blurred_noise = noise_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return blurred_noise

def create_background(size, background_type='white'):
    """
    Create background based on specified type
    """
    if background_type == 'white':
        return Image.new('RGB', size, 'white')
    elif background_type == 'solid':
        return generate_solid_background(size)
    elif background_type == 'gradient':
        return generate_gradient_background(size, 
                                         direction=random.choice(['horizontal', 'vertical', 'diagonal']))
    elif background_type == 'noise':
        return generate_noise_background(size)
    elif background_type == 'blurred_noise':
        return generate_blurred_noise_background(size)
    elif background_type == 'clouds':
        return generate_clouds_background(size)
    else:
        return Image.new('RGB', size, 'white')

def remove_white_background(image):
    """
    Remove white background from image by converting it to transparency
    
    Args:
        image: PIL Image
    Returns:
        PIL Image with transparency
    """
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Get image data
    data = np.array(image)
    
    # Create mask for white pixels (with tolerance)
    tolerance = 30  # Zwiększona tolerancja dla białego koloru
    r, g, b, a = data.T
    white_areas = (r > 255-tolerance) & (g > 255-tolerance) & (b > 255-tolerance)
    
    # Convert white pixels to transparent
    data[..., 3] = np.where(white_areas.T, 0, 255)
    
    return Image.fromarray(data)

def blend_images(background, foreground):
    """
    Blend foreground image with background using simple alpha compositing
    
    Args:
        background: PIL Image (RGB)
        foreground: PIL Image (RGBA)
    Returns:
        PIL Image
    """
    # Convert images to numpy arrays
    bg = np.array(background).astype(float)
    fg = np.array(foreground).astype(float)
    
    # Extract alpha channel and normalize
    alpha = fg[..., 3:] / 255.0
    
    # Simple alpha compositing: result = alpha * foreground + (1 - alpha) * background
    result_rgb = (fg[..., :3] * alpha + bg[..., :3] * (1 - alpha)).astype(np.uint8)
    
    # Create final image
    result = np.dstack((result_rgb, np.full_like(result_rgb[..., 0], 255)))
    
    return Image.fromarray(result)

def create_diverse_logo_image(logos, target_size=640):
    """
    Create image with 1-3 logos of different sizes
    
    Args:
        logos: List of available logo images
        target_size: Target size for the output image (both width and height)
    """
    # Randomly choose background type
    background_type = random.choices(
        ['white', 'solid', 'gradient', 'noise', 'blurred_noise', 'clouds'],
        weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]  # 50% white, 10% każdy inny typ
    )[0]
    
    # Create background
    background = create_background((target_size, target_size), background_type)
    
    # Randomly decide number of logos (1-3)
    num_logos = random.randint(1, 3)
    
    # Randomly select logos (ensure no duplicates)
    selected_logos = random.sample(logos, num_logos)
    
    # Keep track of placed boxes
    placed_boxes = []
    labels = []
    
    for logo in selected_logos:
        # Convert logo to PIL Image if needed
        if not isinstance(logo, Image.Image):
            logo = Image.fromarray(logo)
        
        # Get original logo size
        logo_width, logo_height = logo.size
        
        # Try to place logo (with multiple attempts)
        max_attempts = 50
        placed = False
        
        for _ in range(max_attempts):
            # Random scale factor (0.1 to 0.3 of image size)
            scale = random.uniform(0.1, 0.3)
            target_width = int(target_size * scale)
            
            # Calculate height maintaining aspect ratio
            target_height = int(target_width * (logo_height / logo_width))
            
            # Skip if logo would be too big
            if target_height >= target_size or target_width >= target_size:
                continue
            
            # Resize logo
            resized_logo = logo.resize((target_width, target_height))
            
            # Remove white background and apply transparency
            processed_logo = remove_white_background(resized_logo)
            
            # Random position (ensure logo fits within image bounds)
            x = random.randint(0, max(0, target_size - target_width))
            y = random.randint(0, max(0, target_size - target_height))
            
            # Find tight bounding box for the processed logo
            bbox = find_tight_bbox(processed_logo)
            if bbox is None:
                continue
                
            # Adjust bbox to the position on background
            x1, y1, x2, y2 = bbox
            actual_box = (x + x1, y + y1, x + x2, y + y2)
            
            if not check_overlap(actual_box, placed_boxes):
                if background_type == 'white':
                    # Dla białego tła po prostu wklejamy logo z przezroczystością
                    background.paste(processed_logo, (x, y), processed_logo)
                else:
                    # Dla innych teł stosujemy mieszanie
                    temp_bg = background.crop((x, y, x + target_width, y + target_height))
                    blended = blend_images(temp_bg, processed_logo)
                    background.paste(blended, (x, y))
                
                placed_boxes.append(actual_box)
                
                # Calculate normalized coordinates (YOLO format)
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x + x1 + box_width/2) / target_size
                center_y = (y + y1 + box_height/2) / target_size
                norm_width = box_width / target_size
                norm_height = box_height / target_size
                
                # Use class ID 1 for logos
                labels.append([1, center_x, center_y, norm_width, norm_height])
                placed = True
                break
        
        if not placed:
            print("Warning: Could not place logo without overlap")
    
    return background, labels

def generate_diverse_dataset(num_samples=2000, output_dir='data/diverse_logos_dataset', max_logos=1000, image_size=640):
    """Generate dataset with diverse logo placements"""
    print(f"Loading logo dataset...")
    dataset = load_dataset("iamkaikai/amazing_logos_v4", split="train")
    
    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'valid' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'valid' / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Convert all images to PIL Images
    print("Processing logos...")
    available_logos = []
    
    # Randomly select indices to limit the number of logos
    total_logos = len(dataset)
    selected_indices = random.sample(range(total_logos), min(max_logos, total_logos))
    
    for i in tqdm(selected_indices, desc="Loading logos", unit="logo"):
        try:
            item = dataset[i]
            if isinstance(item['image'], Image.Image):
                logo = item['image'].copy()
                # Convert to RGB to ensure consistency and reduce memory
                if logo.mode != 'RGB':
                    logo = logo.convert('RGB')
                # Resize large logos to reduce memory usage
                if logo.size[0] > 1000 or logo.size[1] > 1000:
                    ratio = min(1000/logo.size[0], 1000/logo.size[1])
                    new_size = (int(logo.size[0] * ratio), int(logo.size[1] * ratio))
                    logo = logo.resize(new_size, Image.LANCZOS)
                available_logos.append(logo)
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(available_logos)} logos")
    if len(available_logos) == 0:
        raise ValueError("No logos were successfully processed!")
    
    # Split into training and validation
    train_samples = int(num_samples * 0.8)  # 80% for training
    val_samples = num_samples - train_samples
    
    # Generate training images
    print(f"\nGenerating {train_samples} training images...")
    for i in tqdm(range(train_samples), desc="Generating training images", unit="img"):
        image, labels = create_diverse_logo_image(available_logos, target_size=image_size)
        
        # Save image
        image_path = output_dir / 'train' / 'images' / f'logo_{i:05d}.jpg'
        image.save(str(image_path), 'JPEG', quality=90)
        
        # Save labels
        label_path = output_dir / 'train' / 'labels' / f'logo_{i:05d}.txt'
        with open(str(label_path), 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Generate validation images
    print(f"\nGenerating {val_samples} validation images...")
    for i in tqdm(range(val_samples), desc="Generating validation images", unit="img"):
        image, labels = create_diverse_logo_image(available_logos, target_size=image_size)
        
        # Save image
        image_path = output_dir / 'valid' / 'images' / f'logo_{i:05d}.jpg'
        image.save(str(image_path), 'JPEG', quality=90)
        
        # Save labels
        label_path = output_dir / 'valid' / 'labels' / f'logo_{i:05d}.txt'
        with open(str(label_path), 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    print(f"\nDataset generated successfully!")
    print(f"Training images: {train_samples}")
    print(f"Validation images: {val_samples}")
    print(f"Dataset saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate diverse logo dataset')
    parser.add_argument('--num_samples', type=int, default=2000,
                      help='Total number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/diverse_logos_dataset',
                      help='Output directory for the dataset')
    parser.add_argument('--max_logos', type=int, default=1000,
                      help='Maximum number of logos to load (default: 1000)')
    parser.add_argument('--image_size', type=int, default=640,
                      help='Size of output images (default: 640)')
    
    args = parser.parse_args()
    
    generate_diverse_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        max_logos=args.max_logos,
        image_size=args.image_size
    )

if __name__ == '__main__':
    main() 