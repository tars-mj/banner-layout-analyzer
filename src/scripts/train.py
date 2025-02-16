from ultralytics import YOLO
import argparse
from pathlib import Path
from src.utils.metrics import analyze_training_results
import wandb
import json
import pandas as pd

def prepare_combined_data_yaml(dataset_type: str, existing_yaml: Path = None) -> str:
    """
    Get data.yaml configuration for training
    """
    # Define paths for dataset configurations
    dataset_yamls = {
        'faces': 'data/faces_dataset/data.yaml',
        'logos': 'data/logos_dataset/data.yaml',
        'diverse_logos': 'data/diverse_logos_dataset/data.yaml',
        'combined': 'data/combined_dataset/data.yaml'
    }
    
    yaml_path = Path(dataset_yamls[dataset_type])
    
    # For combined dataset, check if it's already prepared
    if dataset_type == 'combined':
        if not yaml_path.exists():
            print("\nError: Combined dataset configuration not found!")
            print("Please prepare the combined dataset first using:")
            print("python src/scripts/prepare_datasets.py --dataset combined")
            raise FileNotFoundError(f"Combined dataset configuration not found at: {yaml_path}")
            
        print("\nUsing existing combined dataset configuration")
        return str(yaml_path)
    
    # For other datasets, check if they exist
    if not yaml_path.exists():
        print(f"\nError: Dataset configuration not found for {dataset_type}!")
        print(f"Please prepare the dataset first using:")
        print(f"python src/scripts/prepare_datasets.py --dataset {dataset_type}")
        raise FileNotFoundError(f"Dataset configuration not found at: {yaml_path}")
    
    return str(yaml_path)

def log_metrics_to_wandb(metrics_file: Path):
    """
    Log training metrics from results.csv to W&B
    """
    if not metrics_file.exists():
        return
        
    try:
        # Read results
        with open(metrics_file) as f:
            results = json.load(f)
            
        # Log metrics
        wandb.log({
            "box_loss": results.get("train/box_loss", 0),
            "cls_loss": results.get("train/cls_loss", 0),
            "dfl_loss": results.get("train/dfl_loss", 0),
            "precision": results.get("metrics/precision(B)", 0),
            "recall": results.get("metrics/recall(B)", 0),
            "mAP50": results.get("metrics/mAP50(B)", 0),
            "mAP50-95": results.get("metrics/mAP50-95(B)", 0),
            "val_box_loss": results.get("val/box_loss", 0),
            "val_cls_loss": results.get("val/cls_loss", 0),
            "val_dfl_loss": results.get("val/dfl_loss", 0)
        })
        
        # Log validation images if available
        if "val_images" in results:
            wandb.log({"validation_predictions": [wandb.Image(img) for img in results["val_images"]]})
            
    except Exception as e:
        print(f"Warning: Could not log metrics to W&B: {str(e)}")

def send_metrics_to_wandb(output_dir: Path):
    """
    Send training metrics to W&B from results.csv
    Can be called after training interruption
    """
    if not wandb.run:
        print("No active W&B run found")
        return

    results_file = output_dir / 'results.csv'
    if not results_file.exists():
        print(f"No results file found at {results_file}")
        return

    print("\nSending metrics to W&B...")
    
    # Read results.csv
    df = pd.read_csv(results_file)
    
    # Log each epoch's metrics
    for index, row in df.iterrows():
        epoch_metrics = {
            'epoch': index,
            'train/box_loss': row['train/box_loss'],
            'train/cls_loss': row['train/cls_loss'],
            'train/dfl_loss': row['train/dfl_loss'],
            'metrics/precision': row['metrics/precision(B)'],
            'metrics/recall': row['metrics/recall(B)'],
            'metrics/mAP50': row['metrics/mAP50(B)'],
            'metrics/mAP50-95': row['metrics/mAP50-95(B)'],
            'val/box_loss': row['val/box_loss'],
            'val/cls_loss': row['val/cls_loss'],
            'val/dfl_loss': row['val/dfl_loss']
        }
        wandb.log(epoch_metrics)
    
    # Log final metrics to summary
    final_metrics = df.iloc[-1].to_dict()
    for key, value in final_metrics.items():
        wandb.run.summary[f"training_metrics/{key}"] = value
    
    # Log training plots
    plots_dir = output_dir / 'plots'
    if plots_dir.exists():
        for plot_file in plots_dir.glob('*.png'):
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
    
    # Log validation predictions
    val_pred_file = output_dir / 'val_batch0_pred.jpg'
    if val_pred_file.exists():
        wandb.log({"validation/predictions": wandb.Image(str(val_pred_file))})

    print("Metrics sent to W&B successfully")

def train_model(
    dataset_type: str,
    resume: bool = False,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    use_wandb: bool = False,
    fraction: float = 1.0,
    name: str = 'combined_detector',
    model_type: str = 'yolov8n.pt'
):
    """
    Train object detection model
    
    Args:
        dataset_type: Type of dataset to use ('faces' or 'logos')
        resume: Whether to use weights from previous training
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        use_wandb: Whether to use Weights & Biases for logging
        fraction: Fraction of dataset to use (0.0-1.0)
        name: Name for this training run
        model_type: Type of YOLO model to use (default: 'yolov8n.pt')
    """
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="banner-layout-analyzer",
            name=f"{name}",
            config={
                "architecture": model_type,
                "dataset": dataset_type,
                "epochs": epochs,
                "image_size": imgsz,
                "batch_size": batch,
                "resume": resume
            }
        )
    
    # Load model - either new or existing
    if resume:
        # Use best weights from previous training as starting point
        model = YOLO(f'runs/detect/{name}/weights/best.pt')
        print("\nUsing weights from previous training as starting point...")
    else:
        # Start from pretrained YOLO model
        model = YOLO(model_type)
        print(f"\nStarting new training with pretrained {model_type} weights...")
    
    # Set output directory
    output_dir = Path(f'runs/detect/{name}')
    
    # Prepare combined dataset configuration
    data_yaml = prepare_combined_data_yaml(dataset_type, existing_yaml=output_dir/'data.yaml' if resume else None)
    
    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        save=True,
        project='runs/detect',
        name=name,
        exist_ok=True,
        plots=True,
        save_period=1,
        val=True,
        device='',
        verbose=True,
        fraction=fraction,
        resume=resume
    )
    
    # Log training metrics from results.csv if using W&B
    if use_wandb:
        results_file = output_dir / 'results.csv'
        if results_file.exists():
            # Read results.csv
            df = pd.read_csv(results_file)
            
            # Log each epoch's metrics
            for index, row in df.iterrows():
                epoch_metrics = {
                    'epoch': index,
                    'train/box_loss': row['train/box_loss'],
                    'train/cls_loss': row['train/cls_loss'],
                    'train/dfl_loss': row['train/dfl_loss'],
                    'metrics/precision': row['metrics/precision(B)'],
                    'metrics/recall': row['metrics/recall(B)'],
                    'metrics/mAP50': row['metrics/mAP50(B)'],
                    'metrics/mAP50-95': row['metrics/mAP50-95(B)'],
                    'val/box_loss': row['val/box_loss'],
                    'val/cls_loss': row['val/cls_loss'],
                    'val/dfl_loss': row['val/dfl_loss']
                }
                wandb.log(epoch_metrics)
            
            # Log final metrics to summary
            final_metrics = df.iloc[-1].to_dict()
            for key, value in final_metrics.items():
                wandb.run.summary[f"training_metrics/{key}"] = value
            
            # Log training plots
            plots_dir = output_dir / 'plots'
            if plots_dir.exists():
                for plot_file in plots_dir.glob('*.png'):
                    wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
            
            # Log validation predictions
            val_pred_file = output_dir / 'val_batch0_pred.jpg'
            if val_pred_file.exists():
                wandb.log({"validation/predictions": wandb.Image(str(val_pred_file))})
    
    # Analyze training results
    print("\nAnalyzing training results...")
    analyze_training_results(
        model_dir=output_dir,
        output_dir=output_dir / 'analysis'
    )
    
    # Close wandb run if it was used
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--dataset', type=str, required=True, 
                      choices=['faces', 'logos', 'diverse_logos', 'combined'],
                      help='Which dataset to use for training (faces, logos, diverse_logos, or combined)')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--fraction', type=float, default=1.0, 
                      help='Fraction of dataset to use (0.0-1.0). Example: 0.5 for 50% of images')
    parser.add_argument('--name', type=str, default='combined_detector',
                      help='Name for this training run (used for output directory)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                      help='YOLO model type to use')
    parser.add_argument('--send-metrics', action='store_true',
                      help='Send metrics to W&B from interrupted training')
    
    args = parser.parse_args()
    
    if args.send_metrics:
        output_dir = Path(f'runs/detect/{args.name}')
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}")
            exit(1)
        send_metrics_to_wandb(output_dir)
        exit(0)

    if args.fraction <= 0.0 or args.fraction > 1.0:
        parser.error("--fraction must be between 0.0 and 1.0")
    
    print("\nTraining Configuration")
    print("=====================")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Resume' if args.resume else 'New training'}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Dataset fraction: {args.fraction*100}%")
    print(f"Run name: {args.name}")
    print(f"W&B logging: {'Enabled' if args.wandb else 'Disabled'}")
    print("=====================\n")
    
    train_model(
        dataset_type=args.dataset,
        resume=args.resume,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        use_wandb=args.wandb,
        fraction=args.fraction,
        name=args.name,
        model_type=args.model
    ) 