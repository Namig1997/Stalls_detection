from subprocess import call
import os
import time
import argparse
import datetime

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.networks import G_Unet_add_all3D, DensePredictor, EncoderCNN

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="nano_train_set", help="name of the dataset")
parser.add_argument("--logs", default=None, help="txt file for logs")
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument("--size", type=int, default=None, help="image size")
opt = parser.parse_args()

os.makedirs("logs/", exist_ok=True)
trained_models = "trained_models/"

if opt.logs is None:
    raise ValueError("define txt file for logging")
if opt.size is None:
    raise ValueError("specify the size of your images")

log_file_header = "epoch,epochs,batch,batches,lr,loss"

call(["echo " + log_file_header + " >> " + "logs/" + opt.logs], shell=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#Unet3D = G_Unet_add_all3D(nz=0)
E = EncoderCNN(feature_dim=64)
Predictor = DensePredictor(64)

if cuda:
    E = E.cuda()
    Predictor = Predictor.cuda()

E.apply(weights_init_normal)
Predictor.apply(weights_init_normal)

E_optimizer = Adam(E.parameters(), lr=opt.lr)
Predictor_optimizer = Adam(Predictor.parameters(), lr=opt.lr) 
lr = opt.lr

BCE = BCELoss()

data = torch.load("../res/data/%s.pth" % opt.dataset_name)


for ep in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(data):
        
        E_optimizer.zero_grad()
        Predictor_optimizer.zero_grad()

        scans = Variable(batch[0].type(Tensor))
        ave_scan = Variable(batch[1].type(Tensor))
        targets = torch.unsqueeze(Variable(batch[2].type(Tensor)), -1)

        flatten_features = E(scans)
        predictions = Predictor(flatten_features)

        L = BCE(predictions, targets)

        L.backward()
        E_optimizer.step()
    
        if (i % 4) == 0:

            statistics = (
                    ep,
                    opt.n_epochs,
                    i,
                    len(data),
                    lr,
                    L.item()
                )

            COMMAND = "\r[Epoch %d/%d] [Batch %d/%d] [lr: %f] [loss: %f]" % tuple(statistics)

            call("echo " + COMMAND, shell=True)

            string_statistics = ",".join(map(lambda x: str(x), statistics))
            call("echo " + string_statistics + " >> " + "logs/" + opt.logs, shell=True)


torch.save(E.state_dict(), trained_models + "E.pth")
torch.save(Predictor.state_dict(), trained_models + "DensePredictor.pth")