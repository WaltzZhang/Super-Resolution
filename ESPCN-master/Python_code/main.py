from __future__ import print_function
import argparse

import os
from os import makedirs
from os.path import exists, join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from train_model import Train_model
from test_model import Test_model
from dataset import DatasetFromFolder
from generate_training_set import generate_training_set

# Settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
print(opt)
factor = opt.upscale_factor

if opt.cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, plese run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

#Train_sets = ["Timofte91", 	"ILSVRC2012_img_val"]
#Train_type = ["bmp", 		"JPEG"]

Test_sets = ["Set5  ", 	"Set14  ", 	"BSDS100"]
Test_type = ["bmp", 	"bmp", 		"jpg"]
Test_dirs = [join("../Test", x) for x in ["Set5", "Set14", "test"]]

train_set = "Timofte91"
img_type = "bmp"

# train_dir contains the 17*17 sub images after crop, so it starts with '_'
train_dir = "../Subs_{}_{}".format(train_set, factor)
if not exists(train_dir):
	generate_training_set(train_set, train_dir, img_type, factor)

model_save_path = "Model_{}_{}".format(train_set, factor)
if not exists(model_save_path):
	makedirs(model_save_path)

print ("===>> Loading Training Set:")
training_set = DatasetFromFolder(image_dir=train_dir, upscale_factor=factor, is_training=True)
training_dataloader = DataLoader(dataset=training_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

print ("===>> Building Model:")
model = Net(upscale_factor=factor).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

print ("===>> Training:")
# Train_model return with best result(min_loss) and its position.
min_loss, position = Train_model(model, criterion, optimizer, training_dataloader, opt.lr, opt.nEpochs, device, model_save_path)

print ("===>> Training ended. Best result from epoch_{}:".format(position))
print ("===>> Min. Loss: 	{:.6f}".format(min_loss))
# Print parameters
print ("===>> HyperParameters:")
print ("	upscale_factor:		{}".format(opt.upscale_factor))
print ("	learning_rate:		{}".format(opt.lr))
print ("	batch_size:		{}".format(opt.batch_size))
print ("	nEpochs:		{}".format(opt.nEpochs))
print ("===>> Evaluating Model:")

final_model = torch.load("{}/epoch{}.pth".format(model_save_path, position))

# Test model. Test_model return with avg_psnr and bic_psnr for each Test_set.
for iteration, test_dir in enumerate(Test_dirs, 0):
	test_set = DatasetFromFolder(image_dir=test_dir, upscale_factor=factor, is_training=False)
	testing_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

	avg_psnr, bic_psnr = Test_model(final_model, testing_dataloader, factor, device)
	print("	{}	: 	Avg. PSNR: {:.4f} dB		Bic. PSNR:	{:.4f} dB".format(Test_sets[iteration], avg_psnr, bic_psnr))