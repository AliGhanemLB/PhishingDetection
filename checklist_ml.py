import pandas as pd
import re
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Initialise spellchecker
spell = SpellChecker()

# Function to extract email address from a string (to handle names with emails)
def extract_email(text):
    """Extract the email address from a string containing name and email."""
    if isinstance(text, str):
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        return email_match.group(0) if email_match else ''
    return ''

# Domain Check: compare sender's domain with receiver's domain
def check_domain(sender, receiver):
    """Check if the domain of the sender matches the receiver's (organization's) domain."""
    sender_email = extract_email(sender)
    receiver_email = extract_email(receiver)

    sender_domain = sender_email.split('@')[1] if sender_email else ''
    receiver_domain = receiver_email.split('@')[1] if receiver_email else ''

    if sender_domain == receiver_domain:
        return 1  # Safe (non-phishing)
    else:
        return 0  # Suspicious (phishing)

# Subject Analysis: Check for urgency and spelling errors
def clean_subject(subject):
    """Clean the subject line by removing 'Re:', 'Fw:', and bracketed text."""
    subject = re.sub(r'^(Re:|Fw:|\[.*?\])', '', subject)
    return subject.strip()

def check_subject(subject):
    """Check for urgency in subject and spelling errors."""
    subject = clean_subject(subject)

    urgency_keywords = ['urgent', 'immediately', 'action required', 'important', 'deadline']
    subject_lower = subject.lower()

    urgency_found = any(word in subject_lower for word in urgency_keywords)
    misspelled = spell.unknown(subject.split())
    spelling_errors = 1 if len(misspelled) > 0 else 0

    return urgency_found, spelling_errors

# Body Content Analysis: Check for sensitive info requests and urgency
def check_body(body):
    """Check the email body for sensitive information requests and urgency."""
    body_lower = body.lower()
    sensitive_keywords = ['password', 'bank account', 'social security', 'credit card', 'pin']
    sensitive_info = any(word in body_lower for word in sensitive_keywords)

    urgency_keywords_body = ['act fast', 'limited time', 'urgent', 'deadline']
    urgency_in_body = any(word in body_lower for word in urgency_keywords_body)

    return sensitive_info, urgency_in_body

# URL Count: Check the number of URLs in the email body
def check_urls(body):
    """Check the number of URLs in the email body."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)
    return len(urls)

# Clean Text: Remove special characters and strip spaces
def clean_text(text):
    """Clean the text by removing special characters, converting to lowercase, and stripping spaces."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Feature Extraction: Combine all features into a consistent size vector
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

# Load data
data = pd.read_excel(r'C:\Users\aligh\OneDrive - University of West London\Project_draft0\TestDataset1.xlsx', engine='openpyxl')

# Drop the 'date' and 'urls' columns since they're not needed
data = data.drop(columns=['urls', 'date'], errors='ignore')  # Use 'errors=ignore' to avoid errors if the columns don't exist

# Handle missing values in the 'label' column
data['label'] = data['label'].fillna(0).astype(int)

# Handle missing subject and body fields
data['subject'] = data['subject'].apply(lambda x: x if pd.notnull(x) else "no content")
data['body'] = data['body'].apply(lambda x: x if pd.notnull(x) else "no content")

# Clean text datax
data['subject'] = data['subject'].apply(clean_text)
data['body'] = data['body'].apply(clean_text)

# Initialise TF-IDF vectorizers
tfidf_subject = TfidfVectorizer(max_features=500, stop_words=None)
tfidf_body = TfidfVectorizer(max_features=500, stop_words=None)

# Fit the vectorisers on the entire subject and body columns
tfidf_subject.fit(data['subject'])
tfidf_body.fit(data['body'])

# Extract features
features = extract_features(data, tfidf_subject, tfidf_body)

# Extract labels (assuming 'label' column contains the phishing labels)
labels = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Assuming you've already fitted your vectorizers during training
with open('tfidf_subject.pkl', 'wb') as f:
    joblib.dump(tfidf_subject, f)

with open('tfidf_body.pkl', 'wb') as f:
    joblib.dump(tfidf_body, f)

# Save the trained model to a file
with open('FINAL_trained_model.pkl', 'wb') as model_file:
    joblib.dump(model, model_file)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])

# Add labels for TP, FP, TN, FN outside of the plot
TP = cm[0, 0]  # True Positive
FP = cm[0, 1]  # False Positive
FN = cm[1, 0]  # False Negative
TN = cm[1, 1]  # True Negative

# Display TP, FP, TN, FN outside of the plot (on the figure)
plt.figtext(0.88, 0.6, f'TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}', horizontalalignment='left', fontsize=12, color='black', weight='bold')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Phishing Email Detection (Real Dataset)')
plt.show()
