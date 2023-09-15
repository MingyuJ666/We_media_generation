import torch
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from torch import nn
from transformers import BertModel
import numpy as np
from dataset import Dataset

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=14, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=14)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # train
    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        # tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            # loss
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # accuarcy
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ test -----------


        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            # loop for validation
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

    torch.save(model,"text_classification.pth")


def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')



bbc_text_df = pd.read_csv('text.csv')
df = pd.DataFrame(bbc_text_df)

np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8*len(df)), int(.9*len(df))])

print(len(df_train),len(df_val), len(df_test))

EPOCHS = 100
model = BertClassifier()
LR = 1e-6
train(model, df_train, df_val, LR, EPOCHS)
evaluate(model, df_test)