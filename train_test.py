import argparse
import tqdm
import time

import torch
import torch.nn.functional as F
from Qtransformer import Classifier

import ROOT
from root_numpy import root2array, rec2array
import logging
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from Modules.Plotter import plotROC

def readConfig(configFile="config_qqyy.json"):
    with open("config/" + configFile, "r") as f_obj:
        config = json.load(f_obj)
        logging.info("Signal sample(s) %s", config["signal_file_list"])
        logging.info("Background sample(s) %s", config["bkg_file_list"])
        logging.info("Signal(s) tree name %s", config["signal_tree_name"])
        logging.info("Background(s) tree name %s", config["bkg_tree_name"])
        logging.info("Signal(s) weight variable %s", config["signal_weight_name"])
        logging.info("Background(s) weight variable %s", config["bkg_weight_name"])
        logging.info("Signal(s) variables %s", config["var_signal"])
        logging.info("Background(s) variables %s", config["var_bkg"])

    return config
    
def Trandform(sigArray, bkgArray, rangConst="_0_1"):
    sigConstrain = []
    bkgConstrain = []

    mini1 = np.min(sigArray, axis=0)
    mini2 = np.min(bkgArray, axis=0)
    maxi1 = np.max(sigArray, axis=0)
    maxi2 = np.max(bkgArray, axis=0)
    mini = np.minimum(mini1, mini2)
    maxi = np.maximum(maxi1, maxi2)

    if rangConst == "_0_1":
        sigConstrain = (sigArray - np.min(sigArray, axis=0)) / np.ptp(sigArray, axis=0)
    elif rangConst == "_n1_1":
        sigConstrain = ((2.*(sigArray - mini)/(maxi-mini))-1)
        bkgConstrain = ((2.*(bkgArray - mini)/(maxi-mini))-1)


    return sigConstrain, bkgConstrain
    
def preparingData(confiFile="config_qqyy.json", prossEvent=10000, fraction=0.5, seed=None):
    config = readConfig(confiFile)

    signal_dataset = root2array(
        filenames=config["signal_file_list"],
        treename=config["signal_tree_name"],
        branches=config["var_signal"],
        selection=config["signal_selection"],
        include_weight=False,
        weight_name="weight",
        stop=prossEvent,
    )

    bkg_dataset = root2array(
        filenames=config["bkg_file_list"],
        treename=config["bkg_tree_name"],
        branches=config["var_bkg"],
        selection=config["bkg_selection"],
        include_weight=False,
        weight_name="weight",
        stop=prossEvent,
    )

    signal_dataset = rec2array(signal_dataset)
    bkg_dataset = rec2array(bkg_dataset)

    signal_dataset, bkg_dataset = Trandform(signal_dataset, bkg_dataset, "_n1_1")
    #print("sig:")
    #print(signal_dataset)

    train_size = int(len(signal_dataset) * fraction)
    test_size = int(len(signal_dataset) * fraction)

    X_signal = signal_dataset
    X_background = bkg_dataset

    y_signal = np.ones(X_signal.shape[0])
    y_background = np.ones(X_background.shape[0])
    y_background = 0 * y_background 

    X = np.concatenate([X_signal, X_background], axis=0)
    y = np.concatenate([y_signal, y_background])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=seed
    )

    labels = {1: "S", 0: "B"}

    train_dataset = {
        labels[1]: X_train[y_train == 1],
        labels[0]: X_train[y_train == 0],
    }
    test_dataset = {labels[1]: X_test[y_test == 1], labels[0]: X_test[y_test == 0]}

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_dataset,
        test_dataset,
        X,
        y,
    )
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, train_loader, optimizer, criterion, Num, Type_model):
    epoch_loss = 0
    epoch_acc = 0
    all_labels = []
    all_predictions = []
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        # Get input and label
        inputs, label = batch
        label = torch.tensor([label], dtype=torch.float32)

        # Loss
        predictions = model(inputs)
        loss = criterion(predictions.squeeze(1), label)  # use squeeze(1) match the input shape
        acc = binary_accuracy(predictions.squeeze(1), label)
        #print(acc)
        
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predictions.squeeze(1).cpu().detach().numpy())

        # backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    auc = roc_auc_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plotROC(fpr, tpr, auc, f"ROC_{Num}_train_{Type_model}")
        
    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

def evaluate(model, iterator, criterion, Num, Type_model):
    epoch_loss = 0
    epoch_acc = 0
    all_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs, label = batch
            label = torch.tensor([label], dtype=torch.float32)

            predictions = model(inputs)
            loss = criterion(predictions.squeeze(1), label)
            acc = binary_accuracy(predictions.squeeze(1), label)
            
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.squeeze(1).cpu().detach().numpy())
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    auc = roc_auc_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plotROC(fpr, tpr, auc, f"ROC_{Num}_val_{Type_model}")   
         
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--q_device', default='default.qubit', type=str)      #Fix
    parser.add_argument('-B', '--batch_size', default=32, type=int)                 #Fix for now.
    parser.add_argument('-E', '--n_epochs', default=10, type=int)                   #Fix for now, change after fix all hyperparameters
    parser.add_argument('-C', '--n_classes', default=2, type=int)                   #Fix
    parser.add_argument('-l', '--lr', default=0.001, type=float)                    #Changeable
    parser.add_argument('-v', '--vocab_size', default=6, type=int)
    parser.add_argument('-e', '--embed_dim', default=6, type=int)
    parser.add_argument('-f', '--ffn_dim', default=2048, type=int)                    #hidden layer dimension of feedforward networks. Changeable
    parser.add_argument('-t', '--n_transformer_blocks', default=2, type=int)       #Changeable
    parser.add_argument('-H', '--n_heads', default=2, type=int)
    parser.add_argument('-q', '--n_qubits_transformer', default=0, type=int) #6 for Quantum
    parser.add_argument('-Q', '--n_qubits_ffn', default=0, type=int)         #6 for Quantum
    parser.add_argument('-L', '--n_qlayers', default=1, type=int)            # For Quantum
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float)            #Changeable, but for few events, 0.1 is good
    args = parser.parse_args()


    Num_dataset = 20000
    if args.n_qubits_transformer > 0:
        Type_model = "Quantum"
    else:
        Type_model = "Classical"
    
    train_data, test_data, y_train, y_test, train_dataset, test_dataset, X, y = preparingData(prossEvent = Num_dataset)
    
    print(f'Training examples: {len(train_data)}')
    print(f'Testing examples:  {len(test_data)}')
    
    train_label = torch.tensor(y_train, dtype=torch.float32)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_label = torch.tensor(y_test, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    
    train_dataset = TensorDataset(train_data, train_label)  
    test_dataset = TensorDataset(test_data, test_label)  

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(f'Train_loader:  {train_loader}')
    print(f'Train_loader:  {test_loader}')
    
    model = Classifier(embed_dim=args.embed_dim,
                           num_heads=args.n_heads,
                           num_blocks=args.n_transformer_blocks,
                           num_classes=args.n_classes,
                           vocab_size=args.vocab_size,
                           ffn_dim=args.ffn_dim,
                           n_qubits_transformer=args.n_qubits_transformer,
                           n_qubits_ffn=args.n_qubits_ffn,
                           n_qlayers=args.n_qlayers,
                           dropout=args.dropout_rate,
                           q_device=args.q_device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    print("---3---")

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    if args.n_classes < 3:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    for iepoch in range(args.n_epochs):
        start_time = time.time()

        print(f"Epoch {iepoch+1}/{args.n_epochs}")
        weight_matrix = model.input_linear.weight
        #print(f"weight_matrix: {weight_matrix}")

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, Num_dataset, Type_model)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion, Num_dataset, Type_model)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        
        print(f'Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
