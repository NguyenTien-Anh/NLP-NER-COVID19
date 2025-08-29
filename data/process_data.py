import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data():
    """Load train and validation datasets from HuggingFace"""
    splits = {
        'train': 'data/train-00000-of-00001-8d65b84ed312651e.parquet',
        'validation': 'data/validation-00000-of-00001-3261f75d3aa4eb4f.parquet'
    }

    # Load train dataset
    train_df = pd.read_parquet("hf://datasets/truongpdd/Covid19-NER-Vietnamese/" + splits["train"])
    print(f"Loaded train dataset: {len(train_df)} samples")

    # Load validation dataset
    val_df = pd.read_parquet("hf://datasets/truongpdd/Covid19-NER-Vietnamese/" + splits["validation"])
    print(f"Loaded validation dataset: {len(val_df)} samples")

    # Combine both datasets
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} samples")

    return combined_df

def convert_to_tab_format(df):
    """Convert dataframe to tab-separated format with blank lines between sentences"""
    formatted_data = []

    for idx, row in df.iterrows():
        words = row['words']
        tags = row['tags']

        # Add word-tag pairs for this sentence
        for word, tag in zip(words, tags):
            formatted_data.append(f"{word}\t{tag}")

        # Add blank line to separate sentences
        formatted_data.append("")

    return "\n".join(formatted_data)

def save_data_splits(formatted_data, output_dir="data"):
    """Split data 80-20 and save to files"""
    # Split the formatted data into sentences (separated by double newlines)
    sentences = formatted_data.split("\n\n")
    # Remove any empty sentences
    sentences = [s for s in sentences if s.strip()]

    # Split 80-20
    train_sentences, test_sentences = train_test_split(
        sentences, test_size=0.2, random_state=42
    )

    print(f"Train sentences: {len(train_sentences)}")
    print(f"Test sentences: {len(test_sentences)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save train data
    train_path = os.path.join(output_dir, "covid_train.txt")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(train_sentences))
    print(f"Saved train data to: {train_path}")

    # Save test data
    test_path = os.path.join(output_dir, "covid_test.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(test_sentences))
    print(f"Saved test data to: {test_path}")

def main():
    """Main processing function"""
    print("Starting data processing...")

    # Load data from HuggingFace
    df = load_data()

    # Convert to tab-separated format
    print("Converting to tab-separated format...")
    formatted_data = convert_to_tab_format(df)

    # Split and save data
    print("Splitting and saving data...")
    save_data_splits(formatted_data)

    print("Data processing completed!")

if __name__ == "__main__":
    main()