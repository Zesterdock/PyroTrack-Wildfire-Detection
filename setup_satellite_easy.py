r"""
Quick satellite dataset setup using pre-downloaded Kaggle datasets.
This is the easiest way to get started with satellite data.

Requirements:
- Kaggle account: https://www.kaggle.com/
- Kaggle API token: https://www.kaggle.com/docs/api

Setup:
1. Install Kaggle API: pip install kaggle
2. Download kaggle.json from Kaggle Account Settings
3. Place in: C:\Users\<YourName>\.kaggle\kaggle.json (Windows)
4. Run: python setup_satellite_easy.py

This will download ~1GB of satellite wildfire images and organize them.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def check_kaggle_installed():
    """Check if Kaggle API is installed."""
    try:
        import kaggle
        print("✓ Kaggle API installed")
        return True
    except ImportError:
        print("❌ Kaggle API not installed")
        print("Install with: pip install kaggle")
        return False


def check_kaggle_authenticated():
    """Check if Kaggle is authenticated."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if kaggle_json.exists():
        print(f"✓ Kaggle authenticated: {kaggle_json}")
        return True
    else:
        print(f"❌ Kaggle not authenticated")
        print(f"Expected file: {kaggle_json}")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print(f"4. Move downloaded kaggle.json to: {kaggle_json.parent}")
        return False


def download_dataset(dataset_id: str, output_dir: Path):
    """
    Download a Kaggle dataset.
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., 'username/dataset-name')
        output_dir: Directory to download to
    """
    print(f"\n📥 Downloading {dataset_id}...")
    
    try:
        # Use Kaggle Python API directly instead of CLI
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and unzip
        api.dataset_download_files(
            dataset_id,
            path=str(output_dir),
            unzip=True
        )
        
        print(f"✓ Downloaded to {output_dir}")
        return True
            
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("\nTroubleshooting:")
        print("1. Verify kaggle.json is in the correct location")
        print("2. Check internet connection")
        print("3. Verify dataset exists: https://www.kaggle.com/datasets/" + dataset_id)
        return False


def organize_dataset(source_dir: Path, target_dir: Path):
    """
    Organize downloaded dataset into train/fire and train/no_fire structure.
    
    Args:
        source_dir: Source directory with downloaded files
        target_dir: Target directory (data/satellite)
    """
    print(f"\n📂 Organizing dataset...")
    
    # Run organize_dataset.py
    try:
        cmd = [
            sys.executable,  # python
            'organize_dataset.py',
            str(source_dir),
            '--target', str(target_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"✓ Dataset organized in {target_dir}")
            return True
        else:
            print(f"⚠️  organize_dataset.py not available, organizing manually...")
            # Manual organization fallback
            return organize_manually(source_dir, target_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        return organize_manually(source_dir, target_dir)


def organize_manually(source_dir: Path, target_dir: Path):
    """Manual fallback organization."""
    fire_dir = target_dir / 'train' / 'fire'
    no_fire_dir = target_dir / 'train' / 'no_fire'
    fire_dir.mkdir(parents=True, exist_ok=True)
    no_fire_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already organized with fire/nofire folders
    potential_fire_dirs = [
        source_dir / 'fire',
        source_dir / 'wildfire',
        source_dir / 'Training and Validation' / 'fire',
        source_dir / 'train' / 'fire',
    ]
    potential_nofire_dirs = [
        source_dir / 'nofire',
        source_dir / 'no_fire',
        source_dir / 'not_fire',
        source_dir / 'Training and Validation' / 'nofire',
        source_dir / 'train' / 'nofire',
        source_dir / 'train' / 'no_fire',
    ]
    
    # Find existing organized directories
    found_fire_dir = None
    found_nofire_dir = None
    
    for d in potential_fire_dirs:
        if d.exists() and d.is_dir():
            found_fire_dir = d
            break
    
    for d in potential_nofire_dirs:
        if d.exists() and d.is_dir():
            found_nofire_dir = d
            break
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    fire_count = 0
    no_fire_count = 0
    
    # Copy from pre-organized directories
    if found_fire_dir:
        print(f"Found fire images in: {found_fire_dir}")
        for file in found_fire_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                dest = fire_dir / file.name
                shutil.copy2(file, dest)
                fire_count += 1
    
    if found_nofire_dir:
        print(f"Found no-fire images in: {found_nofire_dir}")
        for file in found_nofire_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                dest = no_fire_dir / file.name
                shutil.copy2(file, dest)
                no_fire_count += 1
    
    # Fallback: search all files and use heuristics
    if fire_count == 0 and no_fire_count == 0:
        print("Searching for images using heuristics...")
        for file in source_dir.rglob('*'):
            if file.is_file() and file.suffix.lower() in image_extensions:
                # Heuristic: files with 'fire' in path go to fire/
                if 'fire' in str(file).lower() and 'nofire' not in str(file).lower():
                    dest = fire_dir / f"fire_{fire_count:04d}{file.suffix}"
                    shutil.copy2(file, dest)
                    fire_count += 1
                else:
                    dest = no_fire_dir / f"no_fire_{no_fire_count:04d}{file.suffix}"
                    shutil.copy2(file, dest)
                    no_fire_count += 1
    
    print(f"✓ Organized {fire_count} fire images, {no_fire_count} no-fire images")
    return fire_count > 0


def verify_dataset(data_dir: Path):
    """Verify the organized dataset."""
    print(f"\n🔍 Verifying dataset...")
    
    fire_dir = data_dir / 'train' / 'fire'
    no_fire_dir = data_dir / 'train' / 'no_fire'
    
    fire_count = len(list(fire_dir.glob('*'))) if fire_dir.exists() else 0
    no_fire_count = len(list(no_fire_dir.glob('*'))) if no_fire_dir.exists() else 0
    
    print(f"  Fire images: {fire_count}")
    print(f"  No-fire images: {no_fire_count}")
    
    if fire_count >= 100 and no_fire_count >= 100:
        print("✓ Dataset looks good!")
        return True
    else:
        print("⚠️  Dataset might be too small (need 100+ images per class)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Easy satellite dataset setup')
    parser.add_argument('--dataset', 
                       default='brsdincer/wildfire-detection-image-data',
                       help='Kaggle dataset ID')
    parser.add_argument('--output', 
                       default='data/satellite',
                       help='Output directory')
    parser.add_argument('--skip-download', 
                       action='store_true',
                       help='Skip download (if already downloaded)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🛰️  EASY SATELLITE DATASET SETUP")
    print("=" * 70)
    
    # Check prerequisites
    if not check_kaggle_installed():
        return
    
    if not check_kaggle_authenticated():
        return
    
    # Create directories
    output_dir = Path(args.output)
    download_dir = Path('downloads')
    download_dir.mkdir(exist_ok=True)
    
    # Download dataset
    if not args.skip_download:
        if not download_dataset(args.dataset, download_dir):
            return
    else:
        print("Skipping download...")
    
    # Find the downloaded dataset directory
    # Kaggle extracts to various folder names, find any subdirectory
    dataset_subdirs = [d for d in download_dir.iterdir() if d.is_dir()]
    
    if not dataset_subdirs:
        print(f"❌ No dataset found in {download_dir}")
        return
    
    # Use the first (or only) subdirectory
    dataset_dir = dataset_subdirs[0]
    print(f"Using dataset directory: {dataset_dir}")
    
    # Organize dataset
    if not organize_dataset(dataset_dir, output_dir):
        print("❌ Failed to organize dataset")
        return
    
    # Verify
    verify_dataset(output_dir)
    
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE!")
    print("=" * 70)
    print(f"Dataset location: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Train perception model: python train_satellite_model.py")
    print("2. Integrate with RL agent:")
    print("   - Edit train_cot_agent.py")
    print("   - Change perception_model_path='models/perception_satellite.pth'")
    print("   - Run: python train_cot_agent.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
