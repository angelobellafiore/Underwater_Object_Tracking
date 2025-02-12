import gdown
import zipfile
import os

# URL of the dataset (Replace with your actual Google Drive file link)
GDRIVE_URL = "https://drive.google.com/drive/folders/1dvwy5C49PWHuvlD5_Cc7hvzU0Urzwm3-"
OUTPUT_ZIP = "dataset.zip"
EXTRACT_FOLDER = "dataset/"


def download_and_extract():
    """Download dataset from Google Drive and extract it."""
    print("Downloading dataset...")
    gdown.download(GDRIVE_URL, OUTPUT_ZIP, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    os.remove(OUTPUT_ZIP)  # Remove the zip file after extraction
    print(f"Dataset extracted to {EXTRACT_FOLDER}")


#if __name__ == "__main__":
#    download_and_extract()

download_and_extract()


#In the terminal, run this script to download and extract the dataset
# python scripts/download_dataset.py