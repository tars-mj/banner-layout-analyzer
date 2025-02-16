import argparse
from src.data.dataset_loader import prepare_face_dataset, prepare_logo_dataset
from pathlib import Path
import os
from huggingface_hub import HfFolder
import shutil
from datasets import config
from src.scripts.generate_diverse_logos import generate_diverse_dataset
import yaml
import random

def get_cache_info():
    """Get information about HuggingFace cache directory"""
    cache_dir = config.HF_DATASETS_CACHE
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            # Get cache size in GB
            cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024**3)
            return {
                'path': str(cache_path),
                'size_gb': cache_size
            }
    return None

def clear_dataset_cache():
    """Clear HuggingFace datasets cache"""
    cache_info = get_cache_info()
    if cache_info:
        print(f"\nClearing cache directory: {cache_info['path']}")
        print(f"Size before clearing: {cache_info['size_gb']:.2f} GB")
        
        try:
            # Manual deletion of cache
            shutil.rmtree(cache_info['path'])
            print("Cache cleared successfully!")
            return True
        except Exception as e:
            print(f"Failed to clear cache: {str(e)}")
            return False
    else:
        print("\nNo cache directory found.")
        return False

def estimate_dataset_size(num_samples: int, image_size: int) -> float:
    """Estimate dataset size in GB based on number of samples and image size"""
    # Assuming 3 channels (RGB), 1 byte per channel
    bytes_per_image = image_size * image_size * 3
    total_bytes = bytes_per_image * num_samples
    # Add 20% for labels and metadata
    total_bytes *= 1.2
    return total_bytes / (1024**3)

def prepare_diverse_logos_dataset(output_dir: str = "data/dataset", num_samples: int = 1000, image_size: int = 640):
    """
    Generate diverse logo detection dataset
    
    Args:
        output_dir: Directory where to save the prepared dataset
        num_samples: Number of images to generate
        image_size: Size to resize images to (both width and height)
    """
    print(f"\nPreparing diverse logos dataset...")
    print(f"Number of samples: {num_samples}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Output directory: {output_dir}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with generating the diverse logos dataset? (y/n): ")
    if response.lower() != 'y':
        print("Generation cancelled by user")
        return
        
    try:
        generate_diverse_dataset(
            num_samples=num_samples,
            output_dir=output_dir,
            max_logos=min(12000, num_samples),  # Use reasonable number of unique logos
            image_size=image_size
        )
        
        print("\nDiverse logos dataset generation completed!")
        print(f"Dataset saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Full error details:", e.__class__.__name__)

def prepare_combined_dataset(output_dir: str = "data", image_size: int = 640):
    """
    Prepare combined dataset by merging faces, logos and diverse_logos datasets
    
    Args:
        output_dir: Base directory containing individual datasets
        image_size: Image size used in datasets
    """
    print("\nPreparing combined dataset...")
    
    # Check if required datasets exist
    base_dir = Path(output_dir)
    faces_dir = base_dir / "faces_dataset"
    logos_dir = base_dir / "logos_dataset"
    diverse_logos_dir = base_dir / "diverse_logos_dataset"
    combined_dir = base_dir / "combined_dataset"
    
    missing_datasets = []
    if not faces_dir.exists():
        missing_datasets.append("faces")
    if not logos_dir.exists():
        missing_datasets.append("logos")
    if not diverse_logos_dir.exists():
        missing_datasets.append("diverse_logos")
    
    if missing_datasets:
        print("\nError: Some required datasets are missing!")
        print("Please prepare the following datasets first:")
        for dataset in missing_datasets:
            print(f"- {dataset}")
        print("\nUse the following commands:")
        print("python src/scripts/prepare_datasets.py --dataset faces --image-size 640 --num_samples 20000")
        print("python src/scripts/prepare_datasets.py --dataset logos --image-size 640 --num_samples 20000")
        print("python src/scripts/prepare_datasets.py --dataset diverse_logos --image-size 640 --num_samples 20000")
        return False
    
    # Create combined dataset directories
    for split in ['train', 'valid']:
        images_dir = combined_dir / split / 'images'
        labels_dir = combined_dir / split / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if source directories exist
        faces_images_dir = faces_dir / split / 'images'
        faces_labels_dir = faces_dir / split / 'labels'
        logos_images_dir = diverse_logos_dir / split / 'images'
        logos_labels_dir = diverse_logos_dir / split / 'labels'
        
        if not all(d.exists() for d in [faces_images_dir, faces_labels_dir, logos_images_dir, logos_labels_dir]):
            print(f"\nError: Some required directories are missing!")
            print(f"Please make sure all datasets are properly prepared.")
            print(f"Missing directories:")
            for d in [faces_images_dir, faces_labels_dir, logos_images_dir, logos_labels_dir]:
                if not d.exists():
                    print(f"- {d}")
            return False
        
        # Get lists of all files
        face_images = sorted(list(faces_images_dir.glob('*')))
        face_labels = sorted(list(faces_labels_dir.glob('*')))
        logo_images = sorted(list(logos_images_dir.glob('*')))
        logo_labels = sorted(list(logos_labels_dir.glob('*')))
        
        # Verify matching pairs
        face_pairs = []
        for img, lbl in zip(face_images, face_labels):
            if img.stem == lbl.stem:
                face_pairs.append((img, lbl, True))  # True oznacza, że to twarz
        
        logo_pairs = []
        for img, lbl in zip(logo_images, logo_labels):
            if img.stem == lbl.stem:
                logo_pairs.append((img, lbl, False))  # False oznacza, że to logo
        
        # Get counts of valid pairs
        num_faces = len(face_pairs)
        num_logos = len(logo_pairs)
        print(f"\nDataset statistics for {split} split:")
        print(f"Number of faces: {num_faces}")
        print(f"Number of logos: {num_logos}")
        
        # Use all faces and twice as many logos (2:1 ratio - logos to faces)
        target_faces = num_faces  # użyj wszystkich twarzy
        target_logos = min(int(num_faces * 2), num_logos)  # użyj 2 razy więcej logo niż twarzy
        
        print(f"Using {target_faces} faces and {target_logos} logos to maintain 2:1 ratio")
        
        # Randomly sample logos if necessary
        if num_logos > target_logos:
            logo_pairs = random.sample(logo_pairs, target_logos)
        
        # Combine and shuffle both datasets
        combined_data = face_pairs + logo_pairs
        random.shuffle(combined_data)  # Losowo mieszamy wszystkie pary
        
        print(f"Copying and shuffling {len(combined_data)} images to {combined_dir}")
        
        # Copy shuffled data
        for idx, (img, lbl, is_face) in enumerate(combined_data):
            # Create new filenames with mixed indexing
            new_prefix = f"sample_{idx:05d}"
            img_ext = img.suffix
            
            # Copy image
            shutil.copy2(img, combined_dir / split / 'images' / f"{new_prefix}{img_ext}")
            
            # Copy and update label
            with open(lbl, 'r') as f:
                labels = f.readlines()
            with open(combined_dir / split / 'labels' / f"{new_prefix}.txt", 'w') as f:
                for label in labels:
                    parts = label.strip().split()
                    if len(parts) >= 5:  # Ensure valid label format
                        parts[0] = '0' if is_face else '1'  # Set class ID (0 for face, 1 for logo)
                        f.write(' '.join(parts) + '\n')
    
    # Create combined dataset configuration
    combined_config = {
        'path': str(combined_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {
            0: 'face',
            1: 'logo'
        }
    }
    
    # Save combined dataset configuration
    config_path = combined_dir / "data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(combined_config, f, sort_keys=False, default_style=None, width=float("inf"))
    
    print("\nCombined dataset preparation completed!")
    print(f"Configuration saved to: {config_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for training')
    parser.add_argument('--dataset', type=str, choices=['faces', 'logos', 'diverse_logos', 'combined'], 
                      help='Which dataset to prepare (faces, logos, diverse_logos, or combined)')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of samples to download/generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Base output directory (default: data)')
    parser.add_argument('--clear-cache', action='store_true',
                      help='Clear HuggingFace datasets cache')
    parser.add_argument('--image-size', type=int, default=640,
                      help='Size to resize images to (default: 640). Use smaller values like 32 for quick experiments')
    parser.add_argument('--quick-test', action='store_true',
                      help='Use quick test settings (32x32 images, 100 samples)')
    
    args = parser.parse_args()
    
    # Apply quick test settings if requested
    if args.quick_test:
        args.image_size = 32
        args.num_samples = 100
        print("\nUsing quick test settings:")
        print("- Image size: 32x32")
        print("- Number of samples: 100")
    
    # Handle cache clearing
    if args.clear_cache:
        clear_dataset_cache()
        if not args.dataset:  # If only clearing cache
            return
    
    if not args.dataset:
        parser.error("--dataset is required unless --clear-cache is used alone")
    
    # Create dataset-specific directory
    if args.dataset == 'faces':
        dataset_dir = Path(args.output_dir) / "faces_dataset"
    elif args.dataset == 'logos':
        dataset_dir = Path(args.output_dir) / "logos_dataset"
    elif args.dataset == 'diverse_logos':
        dataset_dir = Path(args.output_dir) / "diverse_logos_dataset"
    else:  # combined
        dataset_dir = Path(args.output_dir)
    
    # Estimate dataset size (except for diverse_logos and combined which are handled differently)
    if args.dataset not in ['diverse_logos', 'combined']:
        estimated_size = estimate_dataset_size(args.num_samples, args.image_size)
        
        # Print cache information
        cache_info = get_cache_info()
        if cache_info:
            print("\nHuggingFace Cache Information")
            print("============================")
            print(f"Cache directory: {cache_info['path']}")
            print(f"Current cache size: {cache_info['size_gb']:.2f} GB")
            print("============================\n")
        
        print("\nDataset Preparation Tool")
        print("=======================")
        print(f"Dataset: {args.dataset}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Estimated size: {estimated_size:.2f} GB")
        print(f"Output directory: {dataset_dir}")
        print("=======================\n")
        
        # Ask for confirmation if dataset is large
        if estimated_size > 1.0 and not args.quick_test:  # More than 1GB
            response = input(f"Dataset will take approximately {estimated_size:.2f} GB. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled by user")
                return
    
    # Prepare the requested dataset
    if args.dataset == 'faces':
        prepare_face_dataset(
            output_dir=str(dataset_dir),
            num_faces=args.num_samples,
            image_size=args.image_size
        )
    elif args.dataset == 'logos':
        prepare_logo_dataset(
            output_dir=str(dataset_dir),
            num_logos=args.num_samples,
            image_size=args.image_size
        )
    elif args.dataset == 'diverse_logos':
        prepare_diverse_logos_dataset(
            output_dir=str(dataset_dir),
            num_samples=args.num_samples,
            image_size=args.image_size
        )
    else:  # combined
        prepare_combined_dataset(
            output_dir=str(dataset_dir),
            image_size=args.image_size
        )

if __name__ == "__main__":
    main() 