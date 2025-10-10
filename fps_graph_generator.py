import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_fps_data(cv_file, ml_file):
    """
    Load FPS data from CSV files.

    Args:
        cv_file: Path to AprilTags CV CSV file
        ml_file: Path to YOLO ML CSV file

    Returns:
        tuple: (cv_data, ml_data) as pandas DataFrames
    """
    cv_data = pd.read_csv(cv_file)
    ml_data = pd.read_csv(ml_file)
    return cv_data, ml_data


def plot_fps_comparison(cv_data, ml_data, output_dir="fps_comparison_graphs"):
    """
    Generate comparison plots for FPS data.

    Args:
        cv_data: DataFrame with AprilTags CV data
        ml_data: DataFrame with YOLO ML data
        output_dir: Directory to save output graphs
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. FPS over Time (Line Plot)
    plt.figure(figsize=(14, 6))
    plt.plot(
        cv_data["timestamp"],
        cv_data["fps"],
        label="AprilTags (CV)",
        color="#2E86AB",
        linewidth=1.5,
        alpha=0.7,
    )
    plt.plot(
        ml_data["timestamp"],
        ml_data["fps"],
        label="YOLO (ML)",
        color="#A23B72",
        linewidth=1.5,
        alpha=0.7,
    )
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("FPS (Frames Per Second)", fontsize=12)
    plt.title(
        "FPS Comparison: AprilTags vs YOLO Over Time", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fps_over_time.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/fps_over_time.png")
    plt.close()

    # 2. FPS Statistics Bar Chart
    cv_stats = {
        "Mean": cv_data["fps"].mean(),
        "Median": cv_data["fps"].median(),
        "Min": cv_data["fps"].min(),
        "Max": cv_data["fps"].max(),
    }

    ml_stats = {
        "Mean": ml_data["fps"].mean(),
        "Median": ml_data["fps"].median(),
        "Min": ml_data["fps"].min(),
        "Max": ml_data["fps"].max(),
    }

    x = np.arange(len(cv_stats))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(
        x - width / 2,
        list(cv_stats.values()),
        width,
        label="AprilTags (CV)",
        color="#2E86AB",
        alpha=0.8,
    )
    bars2 = plt.bar(
        x + width / 2,
        list(ml_stats.values()),
        width,
        label="YOLO (ML)",
        color="#A23B72",
        alpha=0.8,
    )

    plt.xlabel("Statistic", fontsize=12)
    plt.ylabel("FPS", fontsize=12)
    plt.title(
        "FPS Statistics Comparison: AprilTags vs YOLO", fontsize=14, fontweight="bold"
    )
    plt.xticks(x, list(cv_stats.keys()), fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fps_statistics.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/fps_statistics.png")
    plt.close()

    # 3. FPS Distribution (Histogram)
    plt.figure(figsize=(14, 6))
    plt.hist(
        cv_data["fps"],
        bins=50,
        alpha=0.6,
        label="AprilTags (CV)",
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.5,
    )
    plt.hist(
        ml_data["fps"],
        bins=50,
        alpha=0.6,
        label="YOLO (ML)",
        color="#A23B72",
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("FPS (Frames Per Second)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("FPS Distribution: AprilTags vs YOLO", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fps_distribution.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/fps_distribution.png")
    plt.close()

    # 4. Box Plot Comparison
    plt.figure(figsize=(10, 6))
    data_to_plot = [cv_data["fps"], ml_data["fps"]]
    box = plt.boxplot(
        data_to_plot,
        labels=["AprilTags (CV)", "YOLO (ML)"],
        patch_artist=True,
        showmeans=True,
    )

    # Color the boxes
    colors = ["#2E86AB", "#A23B72"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("FPS (Frames Per Second)", fontsize=12)
    plt.title(
        "FPS Box Plot Comparison: AprilTags vs YOLO", fontsize=14, fontweight="bold"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fps_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/fps_boxplot.png")
    plt.close()

    # 5. Running Average (Smoothed FPS)
    window_size = 30  # 30 frames window
    cv_data["fps_smooth"] = (
        cv_data["fps"].rolling(window=window_size, center=True).mean()
    )
    ml_data["fps_smooth"] = (
        ml_data["fps"].rolling(window=window_size, center=True).mean()
    )

    plt.figure(figsize=(14, 6))
    plt.plot(
        cv_data["timestamp"],
        cv_data["fps_smooth"],
        label="AprilTags (CV) - Smoothed",
        color="#2E86AB",
        linewidth=2,
    )
    plt.plot(
        ml_data["timestamp"],
        ml_data["fps_smooth"],
        label="YOLO (ML) - Smoothed",
        color="#A23B72",
        linewidth=2,
    )
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("FPS (Frames Per Second)", fontsize=12)
    plt.title(
        f"Smoothed FPS Comparison (Window: {window_size} frames)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fps_smoothed.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir}/fps_smoothed.png")
    plt.close()

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("DETAILED FPS STATISTICS")
    print("=" * 60)

    print("\nAprilTags (CV):")
    print(f"  Mean FPS:     {cv_stats['Mean']:.2f}")
    print(f"  Median FPS:   {cv_stats['Median']:.2f}")
    print(f"  Min FPS:      {cv_stats['Min']:.2f}")
    print(f"  Max FPS:      {cv_stats['Max']:.2f}")
    print(f"  Std Dev:      {cv_data['fps'].std():.2f}")
    print(f"  Total Frames: {len(cv_data)}")
    print(f"  Duration:     {cv_data['timestamp'].max():.2f} seconds")

    print("\nYOLO (ML):")
    print(f"  Mean FPS:     {ml_stats['Mean']:.2f}")
    print(f"  Median FPS:   {ml_stats['Median']:.2f}")
    print(f"  Min FPS:      {ml_stats['Min']:.2f}")
    print(f"  Max FPS:      {ml_stats['Max']:.2f}")
    print(f"  Std Dev:      {ml_data['fps'].std():.2f}")
    print(f"  Total Frames: {len(ml_data)}")
    print(f"  Duration:     {ml_data['timestamp'].max():.2f} seconds")

    print("\nPerformance Comparison:")
    fps_diff = cv_stats["Mean"] - ml_stats["Mean"]
    fps_percent = (fps_diff / ml_stats["Mean"]) * 100
    print(f"  FPS Difference: {fps_diff:.2f} FPS ({fps_percent:+.1f}%)")

    if fps_diff > 0:
        print(f"  AprilTags is {fps_percent:.1f}% faster than YOLO")
    else:
        print(f"  YOLO is {-fps_percent:.1f}% faster than AprilTags")

    print("=" * 60)


def main():
    """
    Main function to generate FPS comparison graphs.
    """
    # File paths
    cv_file = "fps_measurement_cv.csv"
    ml_file = "fps_measurement_ml.csv"

    # Check if files exist
    if not Path(cv_file).exists():
        print(f"Error: {cv_file} not found!")
        return

    if not Path(ml_file).exists():
        print(f"Error: {ml_file} not found!")
        return

    print("Loading FPS data...")
    cv_data, ml_data = load_fps_data(cv_file, ml_file)

    print(f"Loaded {len(cv_data)} frames from AprilTags (CV)")
    print(f"Loaded {len(ml_data)} frames from YOLO (ML)")

    print("\nGenerating comparison graphs...")
    plot_fps_comparison(cv_data, ml_data)

    print("\n✓ All graphs generated successfully!")


if __name__ == "__main__":
    main()
