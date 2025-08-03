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
    parser.add_argument('--source', type=str, choices=['dropbox', 'kaggle'], 
                        default='dropbox', help='Download source to use')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # UCF Crime dataset URLs (multiple sources available)
    if args.source == 'dropbox':
        # Dropbox mirror (direct download)
        dataset_url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=1"
        print("Using Dropbox source...")
    else:
        print("Kaggle source requires manual download due to authentication.")
        print("Please visit: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset")
        print("Download the dataset and extract it to the specified data directory.")
        return
    
    # Note: The dataset is very large (several GB)
    
    print("Downloading UCF Crime dataset...")
    print("Note: This is a large dataset (~128 hours of video, several GB)")
    zip_path = os.path.join(args.data_dir, "ucf_crime_dataset.zip")
    
    # Check if dataset already exists (look for common UCF-Crime structure)
    expected_dirs = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
                     'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 
                     'Stealing', 'Vandalism', 'Normal']
    
    existing_dirs = [d for d in expected_dirs if os.path.exists(os.path.join(args.data_dir, d))]
    
    if len(existing_dirs) > 5:  # If more than 5 expected directories exist
        print("UCF Crime dataset appears to already exist. Skipping download.")
        print(f"Found directories: {existing_dirs}")
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
        print("\nAlternative download options:")
        print("1. Official UCF source: https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset")
        print("2. Dropbox mirror: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0")
        print("3. Kaggle dataset: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset")
        print("\nManual instructions:")
        print("1. Download the UCF Crime dataset from one of the sources above")
        print("2. Extract the contents to the './data' directory")
        print("3. Ensure the directory structure has:")
        print("   - Anomaly videos in subdirectories by crime type")
        print("   - Normal videos in a 'Normal' directory")
        print("   - Annotation files (if available)")

if __name__ == "__main__":
    main()
