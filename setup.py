"""
Setup script for Toxic Comment Classifier
Automates initial setup and NLTK downloads
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("="*60)
    print("Installing required packages...")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    print("\n" + "="*60)
    print("Downloading NLTK data...")
    print("="*60)
    
    try:
        import nltk
        
        datasets = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
        
        for dataset in datasets:
            print(f"Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n" + "="*60)
    print("Creating directories...")
    print("="*60)
    
    directories = ['data', 'models']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/ directory")
    
    return True


def generate_sample_data():
    """Generate sample dataset"""
    print("\n" + "="*60)
    print("Generating sample dataset...")
    print("="*60)
    
    response = input("Generate sample dataset? (y/n): ").lower()
    
    if response == 'y':
        try:
            from src.generate_sample_data import generate_sample_dataset
            generate_sample_dataset(n_samples=2000, output_path='data/train.csv')
            return True
        except Exception as e:
            print(f"❌ Error generating sample data: {e}")
            return False
    else:
        print("⚠️  Skipped. You can generate it later with: python src/generate_sample_data.py")
        return True


def main():
    """Main setup function"""
    print("\n")
    print("="*60)
    print("TOXIC COMMENT CLASSIFIER - SETUP")
    print("="*60)
    print("\n")
    
    steps = [
        ("Installing packages", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Creating directories", create_directories),
        ("Generating sample data", generate_sample_data)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Train models: python src/train_models.py")
    print("2. Run web app: streamlit run app.py")
    print("\nFor more information, see QUICKSTART.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
