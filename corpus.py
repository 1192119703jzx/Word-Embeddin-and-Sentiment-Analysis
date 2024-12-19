import torch
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api

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

def pad_seq(seq):
    if len(seq) < max_len:
        seq.extend([0] * (max_len - len(seq)))
    elif len(seq) > max_len:
        seq = seq[:max_len]
    return seq

class W2VDataset(Dataset):
    def __init__(self, mapped_texts, labels):
        self.token_lists = [pad_seq(mt) for mt in mapped_texts]
        self.labels = np.array(labels).reshape(-1, 1)
    def __len__(self):
        return len(self.token_lists)
    def __getitem__(self, idx):        
        return torch.LongTensor(self.token_lists[idx]).cuda(), torch.LongTensor([self.labels[idx]]).squeeze().cuda()

# Load pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

max_len = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_json('train.jsonl', lines=True)
df = df[df['text'].notna()]

df['cleaned_text'] = df['text'].apply(preprocess_text)

texts = df['cleaned_text'].to_list()
texts = [[token for token in text.split() if token in word2vec_model.key_to_index] for text in texts]
mapped_texts = [[word2vec_model.key_to_index[token] for token in tokens_list] for tokens_list in texts]

labels = df['label'].to_list()

f_train, f_rem, l_train, l_rem = train_test_split(mapped_texts, labels, test_size=0.2, random_state=42)

train_dataset = W2VDataset(f_train, l_train)
dev_dataset = W2VDataset(f_rem, l_rem)

# Save the datasets
torch.save(train_dataset, 'train_dataset.pth')
torch.save(dev_dataset, 'dev_dataset.pth')

#preprocess test data
df_test = pd.read_json('test.jsonl', lines=True)
df_test = df_test[df_test['text'].notna()]
df_test['cleaned_text'] = df_test['text'].apply(preprocess_text)
texts_test = df_test['cleaned_text'].to_list()
texts_test = [[token for token in text.split() if token in word2vec_model.key_to_index] for text in texts_test]
mapped_texts_test = [[word2vec_model.key_to_index[token] for token in tokens_list] for tokens_list in texts_test]
labels_test = df_test['label'].to_list()
test_dataset = W2VDataset(mapped_texts_test, labels_test)
torch.save(test_dataset, 'test_dataset.pth')