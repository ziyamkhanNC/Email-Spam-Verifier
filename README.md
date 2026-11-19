# Email Spam Verifier

A machine learning application that classifies emails as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP), TF-IDF feature extraction, and Multinomial Naive Bayes.

## Project Overview

The Email Spam Verifier analyzes an email's subject and message to determine whether it is likely to be spam or legitimate.

The project covers the complete machine learning workflow, including dataset exploration, text preprocessing, feature extraction, model training, evaluation, and application development.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Streamlit
- Matplotlib
- Joblib

## Dataset

**Dataset:** Enron Spam Dataset  
**Source:** Hugging Face, SetFit/enron_spam

Original dataset contains 33,716 email records with the following fields:

- Message ID
- Subject
- Message
- Spam/Ham
- Date

The dataset was processed to remove empty emails, normalize text, and prepare the classification labels.

After preprocessing:
- Records: 30,462
- Duplicate text-label records identified: 325
- Conflicting text labels: 0
- Missing values: 0

After duplicate removal during model preparation, the final dataset contained 30,137 records.

## Project Workflow

### 1. Exploratory Data Analysis

- Dataset structure and dimensions
- Missing-value analysis
- Spam and ham distribution
- Email length analysis
- Data visualization

### 2. Data Preprocessing

- Handling missing subjects and messages
- Combining email subjects and message content
- Removing empty emails
- Removing duplicate records
- Converting text to lowercase
- Removing HTML tags and URLs
- Normalizing whitespace

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert email text into numerical features.

Configuration:
- Maximum features: 50,000
- N-gram range: (1, 2)
- Minimum document frequency: 2
- English stop words enabled

### 4. Model Training

**Algorithm:** Multinomial Naive Bayes

The dataset was divided using an 80/20 stratified train-test split.

- Training records: 24,109
- Testing records: 6,028
- Random state: 42
- Alpha: 0.5

### 5. Model Evaluation

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.00% |
| Precision | 99.00% |
| Recall | 98.93% |
| F1-Score | 98.97% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 3,089 | 29 |
| Spam | 31 | 2,879 |

These results represent performance on the held-out test set.

## Application

The Streamlit application allows users to:

- Enter an email subject.
- Paste an email message.
- Classify the email.
- View the predicted category.
- View prediction confidence.
- Compare spam and ham probabilities.

## Project Structure

```text
Spam Email Verifier/
│
├── data/
│   ├── raw/
│   │   └── enron_spam_data.csv
│   │
│   └── processed/
│       └── cleaned_emails.csv
│
├── models/
│   └── spam_classifier.joblib
│
├── notebooks/
│
├── screenshots/
│   ├── 01_main_application.png
│   ├── 02_spam_result.png
│   └── 03_legitimate_result.png
│
├── app.py
├── explore_dataset.py
├── preprocess_data.py
├── train_model.py
├── capture_screenshots.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Spam Email Verifier"
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Training the Model

To preprocess the dataset:

```bash
python preprocess_data.py
```

To train and evaluate the classifier:

```bash
python train_model.py
```

The trained model is saved in:

```text
models/spam_classifier.joblib
```

## Screenshots

Application screenshots are available in the `screenshots/` directory.

## Limitations

- Classification depends on patterns learned from the training dataset.
- Unfamiliar email formats may produce incorrect predictions.
- Model confidence is an estimated probability, not a guarantee of correctness.
- Evaluation results may differ on emails from other sources.

## Future Improvements

- Expand the training dataset.
- Evaluate performance on an independent external dataset.
- Compare additional classification algorithms.
- Improve text preprocessing.
- Add batch email classification.
- Deploy the application publicly.

## Author

**Muhammad Ziyam Khan**

Email Spam Verifier  
Machine Learning and NLP Project

## License

Add an appropriate open-source license before publicly distributing this project.