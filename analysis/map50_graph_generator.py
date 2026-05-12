import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_BASE = Path(__file__).parent.parent


def load_training_results(csv_file):
    """
    Load training results from YOLO CSV file.
    
    Args:
        csv_file: Path to the results.csv file
    
    Returns:
        pandas DataFrame with training results
    """
    data = pd.read_csv(csv_file)
    return data


def plot_map50(data, output_dir=None):
    if output_dir is None:
        output_dir = str(_BASE / "results" / "training_comparison_graphs")
    """
    Generate mAP@0.5 plot from training results.
    
    Args:
        data: DataFrame with training results
        output_dir: Directory to save output graphs
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Extract epoch and mAP50 data
    epochs = data['epoch']
    map50 = data['metrics/mAP50(B)']
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, map50, color='#2E86AB', linewidth=2.5, marker='o', 
             markersize=4, markerfacecolor='white', markeredgewidth=1.5, 
             markeredgecolor='#2E86AB', label='mAP@0.5')
    
    # Add a trend line
    z = np.polyfit(epochs, map50, 3)  # 3rd degree polynomial
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), "--", color='#F18F01', alpha=0.7, 
             linewidth=2, label='Trend')
    
    # Highlight best mAP50
    best_epoch = epochs[map50.idxmax()]
    best_map50 = map50.max()
    plt.scatter(best_epoch, best_map50, color='#C73E1D', s=200, 
                zorder=5, edgecolors='white', linewidths=2,
                label=f'Best: {best_map50:.4f} @ Epoch {int(best_epoch)}')
    
    # Annotations
    plt.annotate(f'Best mAP@0.5\n{best_map50:.4f}',
                xy=(best_epoch, best_map50),
                xytext=(best_epoch + 5, best_map50 - 0.05),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='black', lw=1.5))
    
    # Labels and title
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('mAP@0.5', fontsize=13, fontweight='bold')
    plt.title('YOLOv8n Training: mAP@0.5 Progress', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits for better visualization
    plt.xlim(epochs.min() - 1, epochs.max() + 1)
    plt.ylim(0, map50.max() + 0.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/map50_yolov8n.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/map50_yolov8n.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("mAP@0.5 TRAINING STATISTICS")
    print("="*60)
    print(f"Total Epochs:       {len(epochs)}")
    print(f"Initial mAP@0.5:    {map50.iloc[0]:.4f} (Epoch {int(epochs.iloc[0])})")
    print(f"Final mAP@0.5:      {map50.iloc[-1]:.4f} (Epoch {int(epochs.iloc[-1])})")
    print(f"Best mAP@0.5:       {best_map50:.4f} (Epoch {int(best_epoch)})")
    print(f"Improvement:        {(best_map50 - map50.iloc[0]):.4f} ({((best_map50 - map50.iloc[0]) / map50.iloc[0] * 100):.1f}%)")
    print(f"Mean mAP@0.5:       {map50.mean():.4f}")
    print(f"Std Dev:            {map50.std():.4f}")
    print("="*60)


def plot_all_metrics(data, output_dir=None):
    if output_dir is None:
        output_dir = str(_BASE / "results" / "training_comparison_graphs")
    """
    Generate comprehensive training metrics plot.
    
    Args:
        data: DataFrame with training results
        output_dir: Directory to save output graphs
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    epochs = data['epoch']
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('YOLOv8n Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # 1. mAP Metrics
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['metrics/mAP50(B)'], color='#2E86AB', 
             linewidth=2, marker='o', markersize=3, label='mAP@0.5')
    ax1.plot(epochs, data['metrics/mAP50-95(B)'], color='#A23B72', 
             linewidth=2, marker='s', markersize=3, label='mAP@0.5-0.95')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('mAP', fontsize=11)
    ax1.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision and Recall
    ax2 = axes[0, 1]
    ax2.plot(epochs, data['metrics/precision(B)'], color='#06A77D', 
             linewidth=2, marker='o', markersize=3, label='Precision')
    ax2.plot(epochs, data['metrics/recall(B)'], color='#F18F01', 
             linewidth=2, marker='s', markersize=3, label='Recall')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Precision and Recall', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Losses
    ax3 = axes[1, 0]
    ax3.plot(epochs, data['train/box_loss'], color='#2E86AB', 
             linewidth=2, label='Box Loss')
    ax3.plot(epochs, data['train/cls_loss'], color='#A23B72', 
             linewidth=2, label='Class Loss')
    ax3.plot(epochs, data['train/dfl_loss'], color='#06A77D', 
             linewidth=2, label='DFL Loss')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Losses', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Losses
    ax4 = axes[1, 1]
    ax4.plot(epochs, data['val/box_loss'], color='#2E86AB', 
             linewidth=2, label='Val Box Loss')
    ax4.plot(epochs, data['val/cls_loss'], color='#A23B72', 
             linewidth=2, label='Val Class Loss')
    ax4.plot(epochs, data['val/dfl_loss'], color='#06A77D', 
             linewidth=2, label='Val DFL Loss')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Validation Losses', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_metrics_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_metrics_overview.png")
    plt.close()


def main():
    """
    Main function to generate mAP@0.5 graph from training results.
    """
    # File path
    csv_file = str(_BASE / "models" / "yolov8n" / "results.csv")
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return
    
    print("Loading training results...")
    data = load_training_results(csv_file)
    
    print(f"Loaded {len(data)} epochs of training data")
    
    print("\nGenerating mAP@0.5 graph...")
    plot_map50(data)
    
    print("\nGenerating comprehensive metrics overview...")
    plot_all_metrics(data)
    
    print("\n✓ All graphs generated successfully!")


if __name__ == "__main__":
    main()
