"""
Train YOLOv8 on Cricket Ball Dataset

Train a custom YOLOv8 model specifically for cricket ball detection.
After training, the model will be saved and can be used in the DRS system.
"""

from ultralytics import YOLO
from pathlib import Path
import os
import torch

# Configuration
DATASET_PATH = Path("cricket_ball_dataset")
MODEL_SIZE = "yolov8n.pt"  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16  # Lower to 8 or 4 if you get memory errors

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else 'cpu'

def check_dataset():
    """Verify dataset structure and files"""
    print("=" * 60)
    print("Checking Dataset...")
    print("=" * 60)
    
    data_yaml = DATASET_PATH / "data.yaml"
    
    if not DATASET_PATH.exists():
        print(f"❌ ERROR: Dataset folder not found: {DATASET_PATH}")
        print("\nPlease extract your Roboflow ZIP file to:")
        print(f"   {DATASET_PATH.absolute()}")
        return False
    
    if not data_yaml.exists():
        print(f"❌ ERROR: data.yaml not found in {DATASET_PATH}")
        print("\nMake sure you extracted the ZIP file correctly.")
        return False
    
    # Check for train/valid folders
    train_images = DATASET_PATH / "train" / "images"
    valid_images = DATASET_PATH / "valid" / "images"
    
    if not train_images.exists():
        print(f"❌ ERROR: Training images not found: {train_images}")
        return False
    
    if not valid_images.exists():
        print(f"❌ ERROR: Validation images not found: {valid_images}")
        return False
    
    # Count images
    train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    valid_count = len(list(valid_images.glob("*.jpg"))) + len(list(valid_images.glob("*.png")))
    
    print(f"✓ Dataset structure verified!")
    print(f"✓ Training images: {train_count}")
    print(f"✓ Validation images: {valid_count}")
    print(f"✓ data.yaml found")
    print()
    
    return True

def train_cricket_ball_detector():
    """Train YOLOv8 on cricket ball dataset"""
    
    print("=" * 60)
    print("YOLOv8 Cricket Ball Detector Training")
    print("=" * 60)
    print()
    
    # Check dataset first
    if not check_dataset():
        print("\n❌ Training aborted. Please fix the dataset issues above.")
        return
    
    data_yaml = DATASET_PATH / "data.yaml"
    
    print("Training Configuration:")
    print(f"  Model: {MODEL_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Dataset: {data_yaml.absolute()}")
    print(f"  Device: {'GPU (CUDA)' if USE_GPU else 'CPU'}")
    if USE_GPU:
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Estimated Time: 15-30 minutes")
    else:
        print(f"  Estimated Time: 2-4 hours")
    print()
    
    print("=" * 60)
    print("Starting Training... (This may take 15-30 min)")
    print("=" * 60)
    print()
    
    try:
        # Load pretrained model
        model = YOLO(MODEL_SIZE)
        
        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=10,  # Early stopping if no improvement for 10 epochs
            save=True,
            project='cricket_models',
            name='cricket_ball_v1',
            verbose=True,
            device=DEVICE  # Auto-detected: GPU if available, else CPU
        )
        
        print("\n" + "=" * 60)
        print("🎉 Training Complete!")
        print("=" * 60)
        print(f"\n✓ Best model saved to: cricket_models/cricket_ball_v1/weights/best.pt")
        print(f"✓ Last model saved to: cricket_models/cricket_ball_v1/weights/last.pt")
        print("\nNext Steps:")
        print("1. Test your model: python test_model.py")
        print("2. Update main.py to use your trained model")
        print("   Replace in main.py:")
        print("   detector = YOLODetector(")
        print("       model_path='cricket_models/cricket_ball_v1/weights/best.pt',")
        print("       confidence_threshold=0.25")
        print("   )")
        print()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        print("\nCommon issues:")
        print("  - Memory error: Try lowering BATCH_SIZE to 8 or 4")
        print("  - CUDA error: Set device='cpu' in train() call")
        return None

if __name__ == "__main__":
    train_cricket_ball_detector()
