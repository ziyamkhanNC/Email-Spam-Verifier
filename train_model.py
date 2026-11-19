
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# File paths
DATA_PATH = Path("data/processed/cleaned_emails.csv")
MODEL_PATH = Path("models/spam_classifier.joblib")


def main():
    print("\n========== SPAM EMAIL MODEL TRAINING ==========\n")

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Basic validation
    df["Email_Text"] = df["Email_Text"].fillna("").astype(str).str.strip()
    df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.lower().str.strip()

    df = df[df["Email_Text"] != ""]
    df = df[df["Spam/Ham"].isin(["spam", "ham"])]

    # Remove duplicate text-label records
    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=["Email_Text", "Spam/Ham"]
    )

    duplicates_removed = before_deduplication - len(df)

    # Identify emails with conflicting labels
    conflicting_texts = (
        df.groupby("Email_Text")["Spam/Ham"]
        .nunique()
    )

    conflicting_texts = conflicting_texts[
        conflicting_texts > 1
    ].index

    print("Duplicate records removed:", duplicates_removed)
    print("Emails with conflicting labels:", len(conflicting_texts))

    # Exclude ambiguous examples
    df = df[~df["Email_Text"].isin(conflicting_texts)]

    print("\nFinal dataset shape:", df.shape)

    print("\nClass distribution:")
    print(df["Spam/Ham"].value_counts())

    # Separate features and labels
    X = df["Email_Text"]
    y = df["Spam/Ham"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Build TF-IDF and Naive Bayes pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])

    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Model training completed.")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=["ham", "spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["ham", "spam"]
        )
    )

    # Save trained pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("\n========== TRAINING COMPLETED ==========\n")


if __name__ == "__main__":
    main()
