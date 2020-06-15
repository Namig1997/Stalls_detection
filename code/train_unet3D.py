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

from models.networks import G_Unet_add_all3D, DensePredictor

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
parser.add_argument("--dataset_name", type=str, default="set_with_labels_and_smiles_8_filtered", help="name of the dataset")
parser.add_argument("--logs", default=None, help="txt file for logs")
parser.add_argument("--size", default=None, help="image size")
opt = parser.parse_args()

os.makedirs("logs/", exist_ok=True)
if opt.logs is None:
    raise ValueError("define txt file for logging")
if opt.size is None:
    raise ValueError("specify the size of your images")

log_file_header = "epoch,epochs,batch,batches,lr,loss"

call(["echo " + log_file_header + " >> " + "/logs/" + opt.logs], shell=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

Unet3D = G_Unet_add_all3D()
Predictor = DensePredictor(opt.size ** 3)

if cuda:
    Unet3D = Unet3D.cuda()
    Predictor = Predictor.cuda()

Unet3D.apply(weights_init_normal)
Predictor.apply(weights_init_normal)

Unet_optimizer = Adam(Unet3D.parameters(), lr=2e-4)
Predictor_optimizer = Adam(Predictor.parameters(), lr=2e-4) 

BCE = BCELoss()

data = torch.load("sets/%s.pt" % opt.dataset_name)


for e in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(data):
        
        Unet_optimizer.zero_grad()
        Predictor_optimizer.zero_grad()

        3d_scan = Variable(batch[0].type(Tensor))
        ave_scan = Variable(batch[1].type(Tensor))
        targets = Variable(batch[2].type(Tensor))

        flatten_features = Unet3D(3d_scan, ave_scan)
        predictions = Predictor(flatten_features)

        L = BCE(predictions, targets)

        L.backward()
    
    if i==30:

        statistics = (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                lr,
                L.item()
            )
        COMMAND = "\r[Epoch %d/%d] [Batch %d/%d] [lr: %f] [loss: %f]" % tuple(statistics)

        call("echo " + COMMAND, shell=True)
