import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

_BASE = Path(__file__).parent.parent

CONFIG = {
    "model_name": "yolo26n.pt",
    "data_yaml": str(_BASE / "dataset" / "data.yaml"),
    "epochs": 100,
    "imgsz": (480, 640),
    "batch": 16,
    "patience": 50,
    "save_period": 10,
    "device": 0 if torch.cuda.is_available() else "cpu",
    "workers": 8,
    "project": str(_BASE / "models"),
    "name": "yolo26n",
}


def check_dataset(data_yaml_path):
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found!")
        return False

    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    print("\nDataset Configuration:")
    print(f"   - Classes: {data['nc']}")
    print(f"   - Names: {data['names']}")
    print(f"   - Train path: {data['train']}")
    print(f"   - Val path: {data['val']}")
    print(f"   - Test path: {data['test']}")

    return True


def train_yolo26n(config):
    print("\n" + "=" * 60)
    print("Starting YOLO26n Fine-tuning")
    print("=" * 60)

    if not check_dataset(config["data_yaml"]):
        return

    device_name = (
        f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    )
    print(f"\nDevice: {device_name}")

    print(f"\nLoading pretrained model: {config['model_name']}")
    model = YOLO(config["model_name"])

    print(f"\nModel Information:")
    print(f"   - Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(
        f"   - Trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}"
    )

    print(f"\nTraining Configuration:")
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

    print(f"\nStarting training...")
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
            # YOLO26 defaults to its MuSGD optimizer via "auto"
            optimizer="auto",
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
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
            label_smoothing=0.0,
            nbs=64,
            dropout=0.0,
            val=True,
            plots=True,
            # Mosaic is the only spatial augmentation enabled.
            # Flips are explicitly disabled because forward/left/right signs
            # are directional — flipping them creates misleading training samples.
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        print(f"\nTraining Results:")
        print(
            f"   - Best epoch: {results.best_epoch if hasattr(results, 'best_epoch') else 'N/A'}"
        )
        print(f"   - Results saved to: {config['project']}/{config['name']}")

        return results

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


def validate_model(config):
    print("\n" + "=" * 60)
    print("Validating Model")
    print("=" * 60)

    best_weights = f"{config['project']}/{config['name']}/weights/best.pt"

    if not os.path.exists(best_weights):
        print(f"Best weights not found at: {best_weights}")
        return

    print(f"\nLoading best weights: {best_weights}")
    model = YOLO(best_weights)

    print("\nRunning validation...")
    metrics = model.val(data=config["data_yaml"])

    print("\nValidation Metrics:")
    print(f"   - mAP50: {metrics.box.map50:.4f}")
    print(f"   - mAP50-95: {metrics.box.map:.4f}")
    print(f"   - Precision: {metrics.box.mp:.4f}")
    print(f"   - Recall: {metrics.box.mr:.4f}")

    return metrics


def test_inference(config, test_image_path=None):
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    best_weights = f"{config['project']}/{config['name']}/weights/best.pt"

    if not os.path.exists(best_weights):
        print(f"Best weights not found at: {best_weights}")
        return

    print(f"\nLoading best weights: {best_weights}")
    model = YOLO(best_weights)

    if test_image_path is None:
        test_dir = _BASE / "dataset" / "test" / "images"
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
                print(f"Using test image: {test_image_path}")

    if test_image_path and os.path.exists(test_image_path):
        print("\nRunning inference...")
        results = model.predict(test_image_path, save=True, conf=0.25)
        print(f"Results saved to: runs/detect/predict/")
        return results
    else:
        print("No test image found")
        return None


def main():
    print("\n" + "=" * 60)
    print("YOLO26n Fine-tuning for Traffic Sign Detection")
    print("=" * 60)

    results = train_yolo26n(CONFIG)

    if results:
        validate_model(CONFIG)
        test_inference(CONFIG)

    print("\n" + "=" * 60)
    print("Process completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
