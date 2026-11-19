import pandas as pd
import re
from pathlib import Path

# File paths
DATA_PATH = Path("data/raw/enron_spam_data.csv")
OUTPUT_PATH = Path("data/processed/cleaned_emails.csv")


def clean_text(text):
    """Clean and normalize email text."""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("\n========== DATA PREPROCESSING ==========\n")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Handle missing values
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    # Combine subject and message
    df["Email_Text"] = df["Subject"] + " " + df["Message"]

    # Remove completely empty emails
    df["Email_Text"] = df["Email_Text"].str.strip()
    df = df[df["Email_Text"] != ""].copy()

    # Remove exact duplicate records
    df = df.drop_duplicates(
        subset=["Email_Text", "Spam/Ham"]
    )

    # Clean email text
    df["Email_Text"] = df["Email_Text"].apply(clean_text)

    # Keep only required columns
    df = df[["Email_Text", "Spam/Ham"]]

    # Create output directory if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Display results
    print("\nProcessed dataset shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate records:", df.duplicated().sum())

    print("\nClass distribution:")
    print(df["Spam/Ham"].value_counts())

    print("\nSaved processed dataset to:", OUTPUT_PATH)
    print("\nPREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()