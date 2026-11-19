
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/raw/enron_spam_data.csv")
OUTPUT_DIR = Path("screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("Columns:", df.columns.tolist())

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

print("\n========== DUPLICATE RECORDS ==========")

print("Duplicate records:", df.duplicated().sum())

print("\n========== CLASS DISTRIBUTION ==========")

class_counts = df["Spam/Ham"].value_counts()

print(class_counts)

print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values)
plt.title("Spam vs Ham Email Distribution")
plt.xlabel("Email Category")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png")
plt.close()

print("\n========== EMAIL LENGTH ANALYSIS ==========")

df["Email_Text"] = (
    df["Subject"].fillna("").astype(str)
    + " "
    + df["Message"].fillna("").astype(str)
)

df["Email_Length"] = df["Email_Text"].str.len()

print("Average email length:",
      round(df["Email_Length"].mean(), 2), "characters")

print("\nEmail length by category:")

print(
    df.groupby("Spam/Ham")["Email_Length"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

plt.figure(figsize=(10, 5))

for category in ["ham", "spam"]:
    lengths = df.loc[
        df["Spam/Ham"] == category, "Email_Length"
    ]

    plt.hist(lengths, bins=50, alpha=0.6, label=category)

display_limit = df["Email_Length"].quantile(0.99)

plt.xlim(0, display_limit)
plt.title("Email Length Distribution")
plt.xlabel("Email Length (Characters)")
plt.ylabel("Number of Emails")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "email_length_distribution.png")
plt.close()

print("\n========== MISSING VALUES ANALYSIS ==========")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

plt.figure(figsize=(8, 5))
plt.bar(missing_values.index, missing_values.values)
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values.png")
plt.close()

print("\n========== FINAL SUMMARY ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Total missing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())
print("Spam emails:", class_counts.get("spam", 0))
print("Ham emails:", class_counts.get("ham", 0))
print("Average email length:",
      round(df["Email_Length"].mean(), 2), "characters")

print("\nCharts saved in:", OUTPUT_DIR)
print("\nEDA COMPLETED")
