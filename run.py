import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from corpus import W2VDataset
from model import FFN_static, LSTMEncoder_static
import gensim.downloader as api
import numpy as np

#hyperparameters
batch_size = 32
num_iter = 20
learning_rate = 0.001

#model parameters
embed_dim = 300
num_hidden = 128

#LSTM parameters
num_layers = 2
biDirectional = True

#Load the datasets
train_dataset = torch.load('train_dataset.pth')
test_dataset = torch.load('test_dataset.pth')
dev_dataset = torch.load('dev_dataset.pth')


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)


word2vec_model = api.load("word2vec-google-news-300")
embedding_matrix = torch.FloatTensor(word2vec_model.vectors)

def run_experiment(model, train_loader, dev_loader, num_iter, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_cache = np.inf
    best_model_state = None

    #training
    for epoch in range(num_iter):
        model.train()
        train_loss = 0
        for token_lists, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(token_lists)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        prediction = []
        with torch.no_grad():
            for token_lists, labels in dev_loader:
                outputs = model(token_lists)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                prediction.append(predicted)
        
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch {epoch+1}/{num_iter}, '
            f'Training Loss: {train_loss/len(train_loader):.4f}, '
            f'Validation Loss: {val_loss/len(dev_loader):.4f}, '
            f'Validation Accuracy: {val_accuracy:.2f}%')

        if val_loss/len(dev_loader) > loss_cache/len(dev_loader) + 0.007:
                print('Early stopping at epoch {}'.format(epoch))
                break
        elif val_loss < loss_cache:
            loss_cache = val_loss
            best_model_state = model.state_dict()


    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


def eval(model, test_loader):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    prediction = []
    with torch.no_grad():
        for token_lists, labels in test_loader:
            outputs = model(token_lists)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            prediction.append(predicted)
    
    val_accuracy = 100 * val_correct / val_total

    print(f'Validation Loss: {val_loss/len(test_loader):.4f}, '
        f'Validation Accuracy: {val_accuracy:.2f}%')


#load the target model
#model = LSTMEncoder_static(embed_dim=embed_dim, hidden_dim=num_hidden, num_classes=3, embedding_matrix=embedding_matrix, num_layers=2, bidirectional=biDirectional).cuda()
#model2 = FFN_static(embed_dim=embed_dim, num_hidden=num_hidden, labels_num=3, embedding_matrix=embedding_matrix).cuda()

model = LSTMEncoder_static(embed_dim=embed_dim, hidden_dim=num_hidden, num_classes=3, embedding_matrix=embedding_matrix, num_layers=2, bidirectional=biDirectional).cuda()
model2 = FFN_static(embed_dim=embed_dim, num_hidden=num_hidden, labels_num=3, embedding_matrix=embedding_matrix).cuda()


# run
run_experiment(model2, train_loader, dev_loader, num_iter, 0.0005)
eval(model2, test_loader)
run_experiment(model, train_loader, dev_loader, num_iter, 0.001)
eval(model, test_loader)




