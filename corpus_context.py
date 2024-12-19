import torch
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Preprocess text function
def preprocess_text(text):
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>|http\S+', '', text)
    # Convert text to lower case
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# TextDataset for tokenizing input
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=10):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = np.array(labels).reshape(-1, 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize a single example
        tokenized_text = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_text, self.labels[idx]

# EmbeddingDataset for storing embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings_list = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        # Return embeddings and labels, move to device
        return self.embeddings_list[idx].to(device), self.labels[idx].to(device)

# Hyperparameters
max_len = 10
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load train and test files
df = pd.read_json('train.jsonl', lines=True)
df = df[df['text'].notna()]
df_test = pd.read_json('test.jsonl', lines=True)
df_test = df_test[df_test['text'].notna()]

# Preprocess train data
df['cleaned_text'] = df['text'].apply(preprocess_text)
labels = df['label'].to_list()
texts = df['cleaned_text'].to_list()

# Preprocess test data
df_test['cleaned_text'] = df_test['text'].apply(preprocess_text)
texts_test = df_test['cleaned_text'].to_list()
labels_test = df_test['label'].to_list()

# Split train data
f_train, f_rem, l_train, l_rem = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_text = TextDataset(f_train, l_train, tokenizer)
dev_text = TextDataset(f_rem, l_rem, tokenizer)
test_text = TextDataset(texts_test, labels_test, tokenizer)

train_loader = DataLoader(train_text, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_text, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_text, batch_size=batch_size, shuffle=True)

# Function to get contextual embeddings
def contextual_training(model, text_loader):
    embeddings = []
    labels = []
    model.eval()  # Set model to evaluation mode
    for batch in text_loader:
        tokenized_texts, batch_labels = batch
        inputs = {key: value.squeeze(1).to(device) for key, value in tokenized_texts.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings
            embeddings.append(cls_embeddings)
            labels.append(batch_labels)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).view(-1, 1)

# Get embeddings for train, dev, and test
train_embed, train_labels = contextual_training(model, train_loader)
dev_embed, dev_labels = contextual_training(model, dev_loader)
test_embed, test_labels = contextual_training(model, test_loader)

# Create EmbeddingDataset
train_set = EmbeddingDataset(train_embed, train_labels)
dev_set = EmbeddingDataset(dev_embed, dev_labels)
test_set = EmbeddingDataset(test_embed, test_labels)

# Save embeddings
torch.save(train_set, 'train_context.pth')
torch.save(dev_set, 'dev_context.pth')
torch.save(test_set, 'test_context.pth')
