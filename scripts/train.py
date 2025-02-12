from ultralytics import YOLO

### PLACEHOLDER
def train():
    """Train a YOLOv8 model on a custom dataset."""

    # Load a pre-trained YOLOv8 model (choose between nano, small, medium, large, xlarge)
    model = YOLO("models/yolov8n.pt")  # You can replace with yolov8m.pt, yolov8l.pt, etc.

    # Train the model on your dataset
    model.train(
        data="dataset/Dataset.yaml",  # Path to your dataset config file
        epochs=50,  # Number of training epochs
        batch=16,  # Adjust based on GPU memory
        imgsz=640,  # Image size for training
        device="cuda"  # Use "cuda" if GPU is available, otherwise "cpu"
    )

    # Save trained model
    model.export(format="torchscript")  # Convert to TorchScript format
    print("Training completed! Model saved.")


#if __name__ == "__main__":
#    train()

#In the terminal, run the script: python scripts/train.py