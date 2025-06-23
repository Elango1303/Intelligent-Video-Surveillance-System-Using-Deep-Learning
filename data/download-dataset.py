import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                   reporthook=t.update_to)

def main():
    parser = argparse.ArgumentParser(description="Download and extract UCF Crime dataset")
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directory where dataset will be stored')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # URL to the UCF Crime dataset
    # Note: You need to replace this with the actual URL for the dataset
    dataset_url = "https://example.com/ucf_crime_dataset.zip"  # Placeholder URL
    
    print("Downloading UCF Crime dataset...")
    zip_path = os.path.join(args.data_dir, "ucf_crime_dataset.zip")
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(args.data_dir, "train")) and \
       os.path.exists(os.path.join(args.data_dir, "test")):
        print("Dataset already exists. Skipping download.")
        return
    
    try:
        # Download the dataset
        download_url(dataset_url, zip_path)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(args.data_dir)
        
        # Remove the zip file to save space
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        print("\nAlternative instructions:")
        print("1. Download the UCF Crime dataset from the official source")
        print("2. Extract the contents to the './data' directory")
        print("3. Ensure the directory structure has 'train' and 'test' folders with class subfolders")

if __name__ == "__main__":
    main()
