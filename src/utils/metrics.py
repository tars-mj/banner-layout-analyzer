import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import os

def load_metrics(results_dir: Path) -> dict:
    """
    Load all available metrics from the training directory
    
    Args:
        results_dir: Directory containing training results
        
    Returns:
        Dictionary containing training history and validation results
    """
    metrics = {}
    
    # Load CSV results if available
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        metrics['training_history'] = pd.read_csv(results_csv)
        
    # Load validation results if available
    val_results = results_dir / "val_results.json"
    if not val_results.exists():
        val_results = results_dir / "results.json"
    
    if val_results.exists():
        with open(val_results) as f:
            metrics['validation'] = json.load(f)
            
    return metrics

def print_training_summary(metrics: dict):
    """
    Print summary of training metrics
    
    Args:
        metrics: Dictionary containing training metrics
    """
    print("\n=== Training Summary ===")
    
    if 'training_history' in metrics:
        history = metrics['training_history']
        print("\nTraining History:")
        print(f"Number of epochs: {len(history)}")
        
        # Print final epoch metrics
        final_epoch = history.iloc[-1]
        print("\nFinal Epoch Metrics:")
        for col in history.columns:
            if col != 'epoch':
                print(f"{col}: {final_epoch[col]:.6f}")
    
    if 'validation' in metrics:
        val = metrics['validation']
        print("\nValidation Results:")
        for metric, value in val.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.6f}")
            else:
                print(f"{metric}: {value}")

def plot_training_curves(metrics: dict, output_dir: Path):
    """
    Plot training curves
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the plot
    """
    if 'training_history' not in metrics:
        return
        
    history = metrics['training_history']
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    loss_columns = [col for col in history.columns if 'loss' in col.lower()]
    for col in loss_columns:
        plt.plot(history['epoch'], history[col], label=col)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(2, 1, 2)
    metric_columns = [col for col in history.columns if 'map' in col.lower()]
    for col in metric_columns:
        plt.plot(history['epoch'], history[col], label=col)
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f'metrics_analysis_{timestamp}.png'
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nMetrics plot saved to: {output_path}")

def export_metrics_report(metrics: dict, output_dir: Path):
    """
    Export detailed metrics report to file
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f'metrics_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("=== Training Metrics Report ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'training_history' in metrics:
            history = metrics['training_history']
            f.write("Training History Summary:\n")
            f.write(f"Number of epochs: {len(history)}\n")
            f.write("\nMetrics progression:\n")
            
            # Write metrics for each epoch
            for _, row in history.iterrows():
                f.write(f"\nEpoch {int(row['epoch'])}:\n")
                for col in history.columns:
                    if col != 'epoch':
                        f.write(f"{col}: {row[col]:.6f}\n")
        
        if 'validation' in metrics:
            val = metrics['validation']
            f.write("\nValidation Results:\n")
            for metric, value in val.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.6f}\n")
                else:
                    f.write(f"{metric}: {value}\n")
    
    print(f"\nDetailed report saved to: {report_path}")

def analyze_training_results(model_dir: Path, output_dir: Path):
    """
    Analyze training results and generate reports
    
    Args:
        model_dir: Directory containing model training results
        output_dir: Directory to save analysis results
    """
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
        
    print(f"\nAnalyzing metrics from: {model_dir}")
    metrics = load_metrics(model_dir)
    
    if not metrics:
        print("No metrics found in the specified directory")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate analysis
    print_training_summary(metrics)
    plot_training_curves(metrics, output_dir)
    export_metrics_report(metrics, output_dir) 