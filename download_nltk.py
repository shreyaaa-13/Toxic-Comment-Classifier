"""Download all required NLTK data"""
import nltk

print("Downloading NLTK data...")

datasets = ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']

for dataset in datasets:
    print(f"Downloading {dataset}...")
    try:
        nltk.download(dataset)
    except Exception as e:
        print(f"Note: {dataset} - {e}")

print("\n✅ NLTK data download complete!")
