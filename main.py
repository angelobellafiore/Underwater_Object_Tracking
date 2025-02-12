import argparse
from scripts.download_dataset import download_and_extract
from scripts.train import train
from scripts.tracking import track_objects


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracking in Underwater Environment")
    subparsers = parser.add_subparsers(dest="command")

    # Download command
    dataset_parser = subparsers.add_parser("dwnld_dataset", help="Download the dataset")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the YOLOv8 model")

    # Track command
    track_parser = subparsers.add_parser("tracking", help="Track objects in a video")
    track_parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    track_parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained YOLOv8 model")
    track_parser.add_argument("--output", type=str, default="results/tracked_output.mp4",
                              help="Path to save output video")

    args = parser.parse_args()

    if args.command == "dwnld_dataset":
        download_and_extract()

    if args.command == "train":
        train()
    elif args.command == "tracking":
        track_objects(args.video, args.model, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#In the terminal, run the script:
# - for downloading the dataset: python main.py dwnld_dataset
# - for training: python main.py train
# - for tracking: python main.py tracking --video path/to/video/input.mp4 --model models/best.pt --output results/tracked_output.mp4