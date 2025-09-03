"""
QuickDraw Dataset Importer
Downloads and organizes QuickDraw data for Text-to-Doodle project
"""
import os
import requests
import numpy as np
from tqdm import tqdm
import json

class QuickDrawImporter:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
        
        # 50 categories for the Text-to-Doodle project
        self.categories = [
            # Animals (20 categories)
            'cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep', 
            'elephant', 'giraffe', 'lion', 'tiger', 'bear', 'rabbit', 
            'mouse', 'frog', 'snake', 'butterfly', 'bee', 'spider',
            
            # Nature & Plants (10 categories)
            'tree', 'flower', 'grass', 'mushroom', 'sun', 'moon', 
            'star', 'cloud', 'mountain', 'rainbow',
            
            # Food (5 categories)
            'apple', 'banana', 'carrot', 'bread', 'cake',
            
            # Transportation (5 categories)
            'car', 'truck', 'bicycle', 'airplane', 'train',
            
            # Objects & Buildings (10 categories)
            'house', 'bridge', 'castle', 'chair', 'table', 
            'book', 'clock', 'key', 'scissors', 'umbrella'
        ]
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Track download status
        self.download_log = {
            'successful': [],
            'failed': [],
            'total_samples': 0
        }
    
    def download_category(self, category, max_samples=5000):
        """Download a specific category from QuickDraw"""
        filename = f"{category}.npy"
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if already exists
        if os.path.exists(filepath):
            print(f"{category}: Already exists, loading...")
            try:
                data = np.load(filepath)
                actual_samples = min(len(data), max_samples)
                print(f"   Loaded {actual_samples} samples")
                self.download_log['successful'].append(category)
                self.download_log['total_samples'] += actual_samples
                return data[:max_samples] if len(data) > max_samples else data
            except Exception as e:
                print(f"   Error loading existing file: {e}")
                os.remove(filepath)  # Remove corrupted file
        
        # Download from web
        url = f"{self.base_url}/{category}.npy"
        
        try:
            print(f"{category}: Downloading...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size == 0:
                    # No content-length header
                    for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                    desc=f"   Downloading {category}"):
                        f.write(chunk)
                else:
                    # With progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"   Downloading {category}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Load and verify the data
            data = np.load(filepath)
            actual_samples = min(len(data), max_samples)
            
            print(f"   Downloaded {len(data)} samples, using {actual_samples}")
            self.download_log['successful'].append(category)
            self.download_log['total_samples'] += actual_samples
            
            return data[:max_samples] if len(data) > max_samples else data
            
        except requests.exceptions.RequestException as e:
            print(f"   Download failed: {e}")
            self.download_log['failed'].append(category)
            return None
        except Exception as e:
            print(f"   Error processing {category}: {e}")
            self.download_log['failed'].append(category)
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up partial download
            return None
    
    def download_all(self, max_samples_per_category=10000):
        """Download all 50 categories"""
        print("Starting QuickDraw Dataset Download")
        print(f"Data directory: {os.path.abspath(self.data_dir)}")
        print(f"Categories: {len(self.categories)}")
        print(f"Max samples per category: {max_samples_per_category}")
        print("=" * 60)
        
        successful_data = {}
        
        for i, category in enumerate(self.categories, 1):
            print(f"[{i:2d}/{len(self.categories)}] {category}")
            
            data = self.download_category(category, max_samples_per_category)
            if data is not None:
                successful_data[category] = data
            
            print()  # Empty line for readability
        
        return successful_data
    
    def create_summary(self):
        """Create a summary of the download"""
        summary = {
            'total_categories': len(self.categories),
            'successful_downloads': len(self.download_log['successful']),
            'failed_downloads': len(self.download_log['failed']),
            'total_samples': self.download_log['total_samples'],
            'successful_categories': self.download_log['successful'],
            'failed_categories': self.download_log['failed']
        }
        
        # Save summary to file
        summary_path = os.path.join(self.data_dir, 'download_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self, summary):
        """Print download summary"""
        print("=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total categories: {summary['total_categories']}")
        print(f"Successful: {summary['successful_downloads']}")
        print(f"Failed: {summary['failed_downloads']}")
        print(f"Total samples: {summary['total_samples']:,}")
        print(f"Data directory: {os.path.abspath(self.data_dir)}")
        
        if summary['failed_categories']:
            print(f"\nFailed categories: {', '.join(summary['failed_categories'])}")
        
        print(f"\nReady categories: {', '.join(summary['successful_categories'])}")


def main():
    """Main function to download QuickDraw data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download QuickDraw Dataset')
    parser.add_argument('--data_dir', default='./data', help='Directory to save data')
    parser.add_argument('--max_samples', type=int, default=10000, 
                       help='Maximum samples per category')
    parser.add_argument('--categories', type=int, default=50,
                       help='Number of categories to download')
    
    args = parser.parse_args()
    
    # Create importer
    importer = QuickDrawImporter(args.data_dir)
    
    # Limit categories if requested
    if args.categories < len(importer.categories):
        importer.categories = importer.categories[:args.categories]
        print(f"Limited to first {args.categories} categories")
    
    try:
        # Download all categories
        successful_data = importer.download_all(args.max_samples)
        
        # Create and print summary
        summary = importer.create_summary()
        importer.print_summary(summary)
        
        # Additional info
        if summary['successful_downloads'] > 0:
            print(f"\nSuccess! Downloaded {summary['successful_downloads']} categories")
            print(f"Use these categories in your Text-to-Doodle model")
            print(f"Data files are in: {os.path.abspath(args.data_dir)}")
        else:
            print(f"\nNo categories downloaded successfully")
            print(f"Check your internet connection and try again")
        
    except KeyboardInterrupt:
        print(f"\nDownload interrupted by user")
    except Exception as e:
        print(f"\nError during download: {e}")
        raise


if __name__ == "__main__":
    main()
