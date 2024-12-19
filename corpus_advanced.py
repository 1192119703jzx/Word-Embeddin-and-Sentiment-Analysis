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
import spacy

def preprocess_text_init(text):
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>|http\S+', '', text)
    # Convert text to lower case
    text = text.lower()
    return text

def preprocess_text_second(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    remaining_words = []
    remaining_indices = []
    
    stemmer = PorterStemmer()
    for idx, word in enumerate(tokens):
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            remaining_words.append(stemmed_word)
            remaining_indices.append(idx)

    cleaned_text = ' '.join(remaining_words)
    return cleaned_text, remaining_indices

# Load pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ratio = 0.5

class AdvancedDataset(Dataset):
    def __init__(self, vectors, labels):
        self.token_lists = vectors
        self.labels = np.array(labels).reshape(-1, 1)
    def __len__(self):
        return len(self.token_lists)
    def __getitem__(self, idx):        
        return torch.FloatTensor(self.token_lists[idx]).to(device), torch.LongTensor([self.labels[idx]]).squeeze().to(device)

def all_in_one(df, ratio):
    df = df[df['text'].notna()]
    df['cleaned_text_init'] = df['text'].apply(preprocess_text_init)

    texts_init = df['cleaned_text_init'].to_list()
    focus_word = []
    for sentence in texts_init:
        keyword = []
        doc = nlp(sentence)
        for idx, token in enumerate(doc):
            if token.pos_ in ["ADV", "VERB", "ADJ"]:
                keyword.append(idx)
        focus_word.append(keyword)

    df[['cleaned_text_second', 'second_index']] = pd.DataFrame(df['cleaned_text_init'].apply(preprocess_text_second).tolist(), index=df.index)

    texts = df['cleaned_text_second'].to_list()
    second_index = df['second_index'].to_list()
    new_texts = []
    new_idx = []
    for idx_1, text in enumerate(texts):
        text_list = []
        index_list = []
        for idx_2, token in enumerate(text.split()):
            if token in word2vec_model.key_to_index:
                text_list.append(token)
                index_list.append(second_index[idx_1][idx_2])
        new_texts.append(text_list)
        new_idx.append(index_list)

    weighted_vector = []
    for i, tokens_list in enumerate(new_texts):
        non_key = np.zeros(word2vec_model.vector_size)
        non_key_count = 0
        key = np.zeros(word2vec_model.vector_size)
        key_count = 0
        for j, token in enumerate(tokens_list):
            word_vector = word2vec_model.get_vector(token)
            if new_idx[i][j] in focus_word[i]:
                key += word_vector
                key_count += 1
            else:
                non_key += word_vector
                non_key_count += 1
        if non_key_count > 0 and key_count > 0:
            non_key_weight = non_key / non_key_count * (1 - ratio)
            key_weight = key / key_count * ratio
        elif non_key_count > 0 and key_count == 0:
            non_key_weight = non_key / non_key_count
            key_weight = key
        elif non_key_count == 0 and key_count > 0:
            non_key_weight = non_key
            key_weight = key / key_count
        weighted_vector.append(non_key_weight + key_weight)
    
    return weighted_vector, df['label'].to_list()

df = pd.read_json('train.jsonl', lines=True)
data_vectors, data_labels = all_in_one(df, ratio)
df_test = pd.read_json('test.jsonl', lines=True)
test_vectors, test_labels = all_in_one(df_test, ratio)

f_train, f_rem, l_train, l_rem = train_test_split(data_vectors, data_labels, test_size=0.2, random_state=42)

train_dataset = AdvancedDataset(f_train, l_train)
dev_dataset = AdvancedDataset(f_rem, l_rem)
test_dataset = AdvancedDataset(test_vectors, test_labels)

# Save the datasets
torch.save(train_dataset, 'train_advanced_2.pth')
torch.save(dev_dataset, 'dev_advanced_2.pth')
torch.save(test_dataset, 'test_advanced_2.pth')