import pandas as pd
import re
import joblib
from sklearn.metrics import classification_report, accuracy_score
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialise spellchecker
spell = SpellChecker()

# Function to extract email address from a string
def extract_email(text):
    """Extract the email address from a string containing name and email."""
    if isinstance(text, str):
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        return email_match.group(0) if email_match else ''
    return ''

# Function to check domain (sender vs receiver)
def check_domain(sender, receiver):
    """Check if the domain of the sender matches the receiver's (organisation's) domain."""
    sender_email = extract_email(sender)
    receiver_email = extract_email(receiver)

    sender_domain = sender_email.split('@')[1] if sender_email else ''
    receiver_domain = receiver_email.split('@')[1] if receiver_email else ''

    if sender_domain == receiver_domain:
        return 1  # Safe (non-phishing)
    else:
        return 0  # Suspicious (phishing)

# Function to clean the subject (remove unnecessary parts)
def clean_subject(subject):
    """Clean the subject line by removing 'Re:', 'Fw:', and bracketed text."""
    subject = re.sub(r'^(Re:|Fw:|\[.*?\])', '', subject)
    return subject.strip()

# Function to check subject (urgency, spelling errors)
def check_subject(subject):
    """Check for urgency in subject and spelling errors."""
    subject = clean_subject(subject)

    urgency_keywords = ['urgent', 'immediately', 'action required', 'important', 'deadline']
    subject_lower = subject.lower()

    urgency_found = any(word in subject_lower for word in urgency_keywords)
    misspelled = spell.unknown(subject.split())
    spelling_errors = 1 if len(misspelled) > 0 else 0

    return urgency_found, spelling_errors

# Function to check body content for sensitive information and urgency
def check_body(body):
    """Check the email body for sensitive information requests and urgency."""
    body_lower = body.lower()
    sensitive_keywords = ['password', 'bank account', 'social security', 'credit card', 'pin']
    sensitive_info = any(word in body_lower for word in sensitive_keywords)

    urgency_keywords_body = ['act fast', 'limited time', 'urgent', 'deadline']
    urgency_in_body = any(word in body_lower for word in urgency_keywords_body)

    return sensitive_info, urgency_in_body

# Function to check the number of URLs in the body
def check_urls(body):
    """Check the number of URLs in the email body."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)
    return len(urls)

# Function to clean text by removing special characters, converting to lowercase, and stripping spaces
def clean_text(text):
    """Clean the text by removing special characters, converting to lowercase, and stripping spaces."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Load the synthetic dataset
synthetic_data = pd.read_excel(r'C:\Users\aligh\OneDrive - University of West London\Project_draft0\GeneratedDataset.xlsx', engine='openpyxl')

# Clean and preprocess the synthetic data
synthetic_data['subject'] = synthetic_data['subject'].apply(lambda x: x if pd.notnull(x) else "no content")
synthetic_data['body'] = synthetic_data['body'].apply(lambda x: x if pd.notnull(x) else "no content")

synthetic_data['subject'] = synthetic_data['subject'].apply(clean_text)
synthetic_data['body'] = synthetic_data['body'].apply(clean_text)

# Load the saved vectorizers
with open('tfidf_subject.pkl', 'rb') as f:
    tfidf_subject = joblib.load(f)

with open('tfidf_body.pkl', 'rb') as f:
    tfidf_body = joblib.load(f)

# Load the saved model
with open('FINAL_trained_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Function to extract features from the synthetic dataset
def extract_features(data, tfidf_subject, tfidf_body):
    features = []

    for index, row in data.iterrows():
        subject = row['subject']
        body = row['body']
        sender = row['sender']
        receiver = row['receiver']

        # Handle empty subject or body
        subject = subject if pd.notnull(subject) and len(subject.strip()) > 0 else "no content"
        body = body if pd.notnull(body) and len(body.strip()) > 0 else "no content"

        # Extract basic features
        domain_check = check_domain(sender, receiver)
        urgency_found, spelling_errors = check_subject(subject)
        sensitive_info, urgency_in_body = check_body(body)
        num_urls = check_urls(body)

        # TF-IDF transformation for subject and body
        subject_tfidf = tfidf_subject.transform([subject]).toarray().flatten()
        body_tfidf = tfidf_body.transform([body]).toarray().flatten()

        # Combine all features into a single vector
        features.append(
            [domain_check, urgency_found, spelling_errors, sensitive_info, urgency_in_body, num_urls] +
            list(subject_tfidf) + list(body_tfidf)
        )

    return features

# Get the features for the synthetic dataset
features_synthetic = extract_features(synthetic_data, tfidf_subject, tfidf_body)

# Make predictions using the loaded model
y_pred_synthetic = model.predict(features_synthetic)

# Assuming the synthetic dataset has a 'label' column for evaluation
labels_synthetic = synthetic_data['label'].values

# Evaluate the model on the synthetic dataset
print("Accuracy on synthetic dataset:", accuracy_score(labels_synthetic, y_pred_synthetic))
print("Classification Report on synthetic dataset:\n", classification_report(labels_synthetic, y_pred_synthetic))

# Confusion matrix for synthetic dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for synthetic dataset
cm_synthetic = confusion_matrix(labels_synthetic, y_pred_synthetic)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(7, 5))
sns.heatmap(cm_synthetic, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])

# Add labels for TP, FP, TN, FN outside of the plot
TP_synthetic = cm_synthetic[0, 0]  # True Positive
FP_synthetic = cm_synthetic[0, 1]  # False Positive
FN_synthetic = cm_synthetic[1, 0]  # False Negative
TN_synthetic = cm_synthetic[1, 1]  # True Negative

# Display TP, FP, TN, FN outside of the plot (on the figure)
plt.figtext(0.88, 0.6, f'TP: {TP_synthetic}\nFP: {FP_synthetic}\nTN: {TN_synthetic}\nFN: {FN_synthetic}', horizontalalignment='left', fontsize=12, color='black', weight='bold')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Phishing Email Detection (Synthetic Dataset)')
plt.show()
