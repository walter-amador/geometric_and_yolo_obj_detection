import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

_BASE = Path(__file__).parent.parent

# Configuration
CONFIG = {
    "model_name": "yolov8n.pt",  # Pretrained YOLOv8 nano model
    "data_yaml": str(_BASE / "dataset" / "data.yaml"),
    "epochs": 100,  # Number of training epochs
    "imgsz": (480, 640),  # Image size for training (matches dataset 640×480)
    "batch": 16,  # Batch size (adjust based on GPU memory)
    "patience": 50,  # Early stopping patience
    "save_period": 10,  # Save checkpoint every N epochs (-1 to disable)
    "device": 0 if torch.cuda.is_available() else "cpu",  # Use GPU if available
    "workers": 8,  # Number of dataloader workers
    "project": str(_BASE / "models"),
    "name": "yolov8n",
}


def check_dataset(data_yaml_path):
    """
    Verify dataset structure and configuration

    Args:
        data_yaml_path: Path to data.yaml file

    Returns:
        bool: True if dataset is valid, False otherwise
    """
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: {data_yaml_path} not found!")
        return False

    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    print("\n📊 Dataset Configuration:")
    print(f"   - Classes: {data['nc']}")
    print(f"   - Names: {data['names']}")
    print(f"   - Train path: {data['train']}")
    print(f"   - Val path: {data['val']}")
    print(f"   - Test path: {data['test']}")

    return True


def train_yolov8n(config):
    """
    Fine-tune YOLOv8n model on custom dataset

    Args:
        config: Dictionary containing training configuration
    """
    print("\n" + "=" * 60)
    print("🚀 Starting YOLOv8n Fine-tuning")
    print("=" * 60)

    # Check if dataset exists and is valid
    if not check_dataset(config["data_yaml"]):
        return

    # Check device
    device_name = (
        f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    )
    print(f"\n💻 Device: {device_name}")

    # Load pretrained YOLOv8n model
    print(f"\n📥 Loading pretrained model: {config['model_name']}")
    model = YOLO(config["model_name"])

    # Display model info
    print(f"\n📋 Model Information:")
    print(f"   - Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(
        f"   - Trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}"
    )

    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Image size: {config['imgsz']}")
    print(f"   - Batch size: {config['batch']}")
    print(f"   - Device: {config['device']}")
    print(f"   - Workers: {config['workers']}")
    print(f"   - Patience: {config['patience']}")
    print(
        f"   - Save period: Every {config['save_period']} epochs"
        if config["save_period"] > 0
        else "   - Save period: Only best and last"
    )

    # Start training
    print(f"\n🏋️  Starting training...")
    print("=" * 60 + "\n")

    try:
        results = model.train(
            data=config["data_yaml"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            patience=config["patience"],
            save=True,
            save_period=config["save_period"],
            device=config["device"],
            workers=config["workers"],
            project=config["project"],
            name=config["name"],
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,  # Keep mosaic until last 10 epochs
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            # Disable all augmentations except mosaic
            mosaic=1.0,  # ✅ Keep mosaic augmentation (probability = 1.0)
            mixup=0.0,  # ❌ Disable mixup
            copy_paste=0.0,  # ❌ Disable copy-paste
            degrees=0.0,  # ❌ Disable rotation
            translate=0.0,  # ❌ Disable translation
            scale=0.0,  # ❌ Disable scaling
            shear=0.0,  # ❌ Disable shear
            perspective=0.0,  # ❌ Disable perspective
            flipud=0.0,  # ❌ Disable vertical flip
            fliplr=0.0,  # ❌ Disable horizontal flip
            hsv_h=0.0,  # ❌ Disable HSV-Hue augmentation
            hsv_s=0.0,  # ❌ Disable HSV-Saturation augmentation
            hsv_v=0.0,  # ❌ Disable HSV-Value augmentation
        )

        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)

        # Print results summary
        print(f"\n📈 Training Results:")
        print(
            f"   - Best epoch: {results.best_epoch if hasattr(results, 'best_epoch') else 'N/A'}"
        )
        print(f"   - Results saved to: {config['project']}/{config['name']}")

        return results

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise


def validate_model(config):
    """
    Validate the trained model on validation set

    Args:
        config: Dictionary containing training configuration
    """
    print("\n" + "=" * 60)
    print("🔍 Validating Model")
    print("=" * 60)

    # Find best weights
    best_weights = f"{config['project']}/{config['name']}/weights/best.pt"

    if not os.path.exists(best_weights):
        print(f"❌ Best weights not found at: {best_weights}")
        return

    print(f"\n📥 Loading best weights: {best_weights}")
    model = YOLO(best_weights)

    # Run validation
    print("\n🔍 Running validation...")
    metrics = model.val(data=config["data_yaml"])

    print("\n📊 Validation Metrics:")
    print(f"   - mAP50: {metrics.box.map50:.4f}")
    print(f"   - mAP50-95: {metrics.box.map:.4f}")
    print(f"   - Precision: {metrics.box.mp:.4f}")
    print(f"   - Recall: {metrics.box.mr:.4f}")

    return metrics


def test_inference(config, test_image_path=None):
    """
    Test inference on a sample image

    Args:
        config: Dictionary containing training configuration
        test_image_path: Path to test image (optional)
    """
    print("\n" + "=" * 60)
    print("🖼️  Testing Inference")
    print("=" * 60)

    # Find best weights
    best_weights = f"{config['project']}/{config['name']}/weights/best.pt"

    if not os.path.exists(best_weights):
        print(f"❌ Best weights not found at: {best_weights}")
        return

    print(f"\n📥 Loading best weights: {best_weights}")
    model = YOLO(best_weights)

    # Use a test image from the dataset if none provided
    if test_image_path is None:
        test_dir = _BASE / "dataset" / "test" / "images"
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
                print(f"📸 Using test image: {test_image_path}")

    if test_image_path and os.path.exists(test_image_path):
        print("\n🔮 Running inference...")
        results = model.predict(test_image_path, save=True, conf=0.25)
        print(f"✅ Results saved to: runs/detect/predict/")
        return results
    else:
        print("❌ No test image found")
        return None


def main():
    """
    Main function to orchestrate the fine-tuning process
    """
    print("\n" + "=" * 60)
    print("🤖 YOLOv8n Fine-tuning for Traffic Sign Detection")
    print("=" * 60)

    # Train the model
    results = train_yolov8n(CONFIG)

    if results:
        # Validate the model
        validate_model(CONFIG)

        # Test inference
        test_inference(CONFIG)

    print("\n" + "=" * 60)
    print("🎉 Process completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
