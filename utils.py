import re

def clean_text(text):
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[\([{}_\=\+\-\*\/\\$%^&#@!~`|<>?:;,\."\']', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text
