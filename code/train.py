from subprocess import call
import os
import time
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score)

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from models.networks import EncoderCNNPredictor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_binary(y, threshold=0.5):
    return y >= threshold

def get_scores(y_true, y_pred, threshold=0.5):
    y_pred_bin = get_binary(y_pred, threshold=threshold)
    scores = {
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "precision": precision_score(y_true, y_pred_bin),
        "recall": recall_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin),
        "auc": roc_auc_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred_bin),
    }
    return scores

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--train_name", type=str, default="micro_train_set", help="name of the train dataset")
parser.add_argument("--test_name", type=str, default="micro_test_set", help="name of the test dataset")
parser.add_argument("--logs", default=None, help="txt file for logs")
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument("--save_models", type=int, default=0, help="Save or not models")
parser.add_argument("--submission", type=str, default="", help="Create or not a submission")
opt = parser.parse_args()

os.makedirs("logs/", exist_ok=True)
trained_models = "trained_models/"

if opt.logs is None:
    raise ValueError("define txt file for logging")

log_file_header = "epoch,epochs,batch,batches,lr,loss"

call(["echo " + log_file_header + " >> " + "logs/" + opt.logs], shell=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

Predictor = EncoderCNNPredictor()

if cuda:
    Predictor = Predictor.cuda()

Predictor_optimizer = Adam(Predictor.parameters(), lr=opt.lr)

decayRate = 0.99
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=Predictor_optimizer, gamma=decayRate)
lr = opt.lr

BCE = BCELoss()

train_set = torch.load("../res/data/%s.pth" % opt.train_name)
test_set = torch.load("../res/data/%s.pth" % opt.test_name)

metrics = pd.DataFrame(None, columns=['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc'])

for ep in range(opt.epoch, opt.n_epochs):
    Predictor.train()
    for i, batch in enumerate(train_set):
        
        Predictor_optimizer.zero_grad()

        scans = Variable(batch[0].type(Tensor))/255.0
        
        ave_scan = Variable(torch.unsqueeze(batch[1].type(Tensor), -1))/255.0
        targets = torch.unsqueeze(Variable(batch[2].type(Tensor)), -1)

        scans = torch.cat((scans, ave_scan), dim=-1)
        predictions = Predictor(scans)

        L = BCE(predictions, targets)

        L.backward()
        Predictor_optimizer.step()
    
        if (i % 4) == 0:

            statistics = (
                    ep,
                    opt.n_epochs,
                    i,
                    len(train_set),
                    lr,
                    L.item()
                )

            COMMAND = "\r[Epoch %d/%d] [Batch %d/%d] [lr: %f] [loss: %f]" % tuple(statistics)

            call("echo " + COMMAND, shell=True)

            string_statistics = ",".join(map(lambda x: str(x), statistics))
            call("echo " + string_statistics + " >> " + "logs/" + opt.logs, shell=True)

    my_lr_scheduler.step()
    lr = Predictor_optimizer.param_groups[0]["lr"]
    #check performance of the model after each epoch
    y_pred = []
    y_true = []
    Predictor.eval()
    for b in test_set:
        
        scans = Variable(b[0].type(Tensor))/255.0
        ave_scan = Variable(torch.unsqueeze(b[1].type(Tensor), -1))/255.0
        scans = torch.cat((scans, ave_scan), dim=-1)
        targets = b[2]

        with torch.no_grad():

            predictions = Predictor(scans)

        
        y_pred.extend(predictions.cpu().numpy().flatten())
        y_true.extend(targets)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    scores = get_scores(y_true, y_pred)
    metrics.loc[ep, ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']] = (scores['accuracy'], scores['precision'],
                                                                                scores['recall'], scores['f1'],
                                                                                scores['auc'], scores['mcc'])

metrics.to_csv("logs/" + "metrics_" + opt.logs)


#use validation set for more training to gain some possible improvements
lr = lr / decayRate
Predictor_optimizer = Adam(Predictor.parameters(), lr=lr)
decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=Predictor_optimizer, gamma=decayRate)
print("additional training on validation set")
for ep in range(opt.n_epochs, opt.n_epochs + opt.n_epochs // 4):
    Predictor.train()
    for i, batch in enumerate(test_set):
        
        Predictor_optimizer.zero_grad()

        scans = Variable(batch[0].type(Tensor))/255.0
        ave_scan = Variable(torch.unsqueeze(batch[1].type(Tensor), -1))/255.0
        scans = torch.cat((scans, ave_scan), dim=-1)
        targets = torch.unsqueeze(Variable(batch[2].type(Tensor)), -1)

        predictions = Predictor(scans)

        L = BCE(predictions, targets)

        L.backward()
        Predictor_optimizer.step()

        if (i % 4) == 0:

            statistics = (
                    ep,
                    opt.n_epochs,
                    i,
                    len(train_set),
                    lr,
                    L.item()
                )

            COMMAND = "\r[Epoch %d/%d] [Batch %d/%d] [lr: %f] [loss: %f]" % tuple(statistics)

            call("echo " + COMMAND, shell=True)

            string_statistics = ",".join(map(lambda x: str(x), statistics))
            call("echo " + string_statistics + " >> " + "logs/" + opt.logs, shell=True)
    
    my_lr_scheduler.step()



if opt.save_models:
    torch.save(Predictor.state_dict(), trained_models + "Predictor.pth")

if opt.submission:
    Predictor.eval()
    submission = pd.read_csv("../res/data/submission_format.csv", index_col=0)
    submission['stalled'] = None

    testset = torch.load("../res/data/test_set.pth")

    for b in testset:

        with torch.no_grad():

            scans = Variable(b[0].type(Tensor))/255.0
            #ave_scan = torch.unsqueeze(Variable(b[1].type(Tensor)), -1).expand(scans.size(0), scans.size(1), scans.size(2), scans.size(3), 20)
            ave_scan = Variable(torch.unsqueeze(b[1].type(Tensor), -1))/255.0
            scans = torch.cat((scans, ave_scan), dim=-1)
            names = b[2]

            predictions = Predictor(scans)

            submission.loc[names, 'stalled'] = predictions.flatten().cpu()
            
    submission['stalled'] = submission['stalled'].apply(lambda x: 1 if x >= 0.5 else 0)
    submission['stalled'] = submission['stalled'].astype(int)
    submission.to_csv("../res/submissions/" + opt.submission)