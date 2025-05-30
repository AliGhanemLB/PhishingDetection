import pandas as pd
import random
import faker
from faker import Faker

# Initialize Faker instance
fake = Faker()

# List of sample domains to add variation
sample_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'example.com', 'company.com', 'company.org']

# Sample subject lines (mixing spammy and non-spammy subjects)
sample_subjects = [
    "Urgent Action Required: Your account has been compromised!",
    "Important Update: Verify your account now",
    "Limited time offer - Act Fast!",
    "Your Amazon Order Status - Action Needed",
    "You’ve won a prize - Claim it Now",
    "Reminder: Your subscription is about to expire",
    "Congratulations! You’ve been selected for an exclusive offer",
    "Final Notice: Your account will be suspended soon",
    "Monthly Newsletter - Company Updates",
    "Scheduled Maintenance for Our Services"
]

# Sample body paragraphs (mixing spam and legitimate content)
sample_bodies = [
    "Dear customer, please be aware that your account has been flagged for suspicious activity. Please confirm your information to secure your account.",
    "Your account is currently inactive. Please log in and verify your credentials to avoid any disruptions to your service.",
    "This is a notification regarding your recent purchase. For any concerns, please feel free to contact our customer service.",
    "We are offering you an exclusive chance to claim your prize. Simply click the link and provide your shipping details.",
    "We’ve noticed some unusual activity on your account. To ensure security, please reset your password by clicking the link below.",
    "Congratulations! You've been selected for a limited-time reward. Don't miss out on this exciting opportunity!",
    "Your account is now at risk of being suspended. Please verify your personal information to avoid any interruption in your services.",
    "This is a reminder about your recent order. If you didn't make this purchase, please contact us immediately.",
    "Be sure to act now! Only 24 hours left to claim your free trial membership.",
    "Dear user, our system will undergo scheduled maintenance on [date]. Please be aware of possible interruptions in service."
]

# Number of records to generate
num_records = 1000

# Function to generate fake emails with varied domains
def generate_email():
    return fake.user_name() + "@" + random.choice(sample_domains)

# Function to generate a random subject
def generate_subject():
    return random.choice(sample_subjects)

# Function to generate a random body
def generate_body():
    return random.choice(sample_bodies)

# Function to generate a random label (0 = non-phishing, 1 = phishing)
def generate_label():
    return random.choice([0, 1])

# Create the dataframe
data = []
for _ in range(num_records):
    row = {
        'sender': generate_email(),
        'receiver': generate_email(),
        'subject': generate_subject(),
        'body': generate_body(),
        'label': generate_label(),
    }
    data.append(row)

# Convert to a pandas dataframe
df_fake = pd.DataFrame(data)



# Save the generated data to a new file
df_fake.to_excel("C:/Users/aligh/OneDrive - University of West London/Project_draft0/GeneratedDataset.xlsx", index=False)

print("Dataset generation complete!")
