import argparse
from pathlib import Path
from src.utils.metrics import analyze_training_results

def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing model training results')
    parser.add_argument('--output_dir', type=str, default='analysis',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_training_results(
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir)
    )

if __name__ == "__main__":
    main() 