import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from corpus_context import EmbeddingDataset
from corpus_advanced import AdvancedDataset
from model_enhanced import FFN_enhanced, LSTMEncoder_enhanced
import numpy as np

#hyperparameters
batch_size = 32
num_iter = 25
learning_rate = 0.0005

#model parameters
num_hidden = 128

#LSTM parameters
num_layers = 2
biDirectional = True


# Load the saved contextual embeddings
train_set = torch.load('train_context.pth')
dev_set = torch.load('dev_context.pth')
test_set = torch.load('test_context.pth')


# Load the saved advanced static embeddings
#train_set = torch.load('train_advanced_2.pth')
#dev_set = torch.load('dev_advanced_2.pth')
#test_set = torch.load('test_advanced_2.pth')


# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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
            labels = labels.squeeze().long()
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
                labels = labels.squeeze().long()
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
        #print(prediction[0])

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
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    prediction = []
    with torch.no_grad():
        for token_lists, labels in test_loader:
            outputs = model(token_lists)
            labels = labels.squeeze().long()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            prediction.append(predicted)
    
    val_accuracy = 100 * val_correct / val_total

    print(f'Validation Loss: {val_loss/len(test_loader):.4f}, '
        f'Validation Accuracy: {val_accuracy:.2f}%')


# for contextual embedding
embed_dim = 768
# for advanced static
#embed_dim = 300

#load the target model
#model = LSTMEncoder_enhanced(embed_dim=embed_dim, hidden_dim=num_hidden, num_classes=3, num_layers=num_layers, bidirectional=biDirectional).cuda()
model = FFN_enhanced(embed_dim=embed_dim, num_hidden=num_hidden, labels_num=3).cuda()

#run
run_experiment(model, train_loader, dev_loader, num_iter, learning_rate)
eval(model, test_loader)

