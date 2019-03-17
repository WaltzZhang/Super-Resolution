from __future__ import print_function
import argparse
import math
from math import log10

import os
from os.path import exists, join
from os import makedirs

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model import Net
from dataset import DatasetFromFolder

# Settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, plese run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

train_dir = "_Timofte91_4.mat"
model_save_path = "Train_Timofte91_4"

Test_dirs = ["mat_Set5", "mat_Set14", "mat_BSDS300"]
Test_sets = ["Set5", "Set14", "BS300"]

if not exists(model_save_path):
	makedirs(model_save_path)

print ("===> Loading Training set:")
train_set = DatasetFromFolder(train_dir, opt.upscale_factor, True)
training_dataloader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
training_size = len(training_dataloader)

print ("===> Bulding model:")
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#scheduler = MultiStepLR(optimizer, milestones=[10, 30, 300], gamma=0.1)

min_loss, position = 1, 0
update_min = False

def PSNR(sr, gt, shave_border=0):
	height, width = gt.shape[:2]
	sr = sr[shave_border:height - shave_border, shave_border:width - shave_border]
	gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
	diff = sr - gt
	rmse = math.sqrt(np.mean(diff ** 2))
	if rmse == 0:
		return 100
	return 20 * math.log10(255.0 / rmse)

def train(epoch):
	epoch_loss = 0
	global min_loss
	global position
	global update_min
	# Calculate Loss
	for batch in training_dataloader:
		input, target = batch[0].to(device), batch[1].to(device)

		optimizer.zero_grad()
		loss = criterion(model(input), target)
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
	epoch_loss /= training_size

	# Update min_loss
	if epoch_loss <= min_loss:
		position, min_loss, update_min = epoch, epoch_loss, True

	# Print loss
	if epoch % 10 == 0:
		print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))

def checkpoint(epoch):
	model_out_patch = "{}/model_epoch_{}.pth".format(model_save_path, epoch)
	torch.save(model, model_out_patch)

# Training
for epoch in range(1, opt.nEpochs + 1):
	#scheduler.step()
	if epoch == 500:
		optimizer.lr = opt.lr / 10
	train(epoch)
	checkpoint(epoch)
	if epoch in [opt.nEpochs/10 * x for x in range(1, 11)]:
		if update_min:
			print ("===> Epoch {} Complete, Min_loss Changed at epoch {}:".format(epoch, position))
			print ("===> Min. Loss: {:.4f}".format(min_loss))
			update_min = False
		else:
			print ("===> Epoch {} Complete, Min_loss Remained at epoch {}:".format(epoch, position))
			print ("===> Min. Loss: {:.4f}".format(min_loss))

# Test on Set5, Set14, BSDS300
print ("===> Training ended. Start testing:")
print ("===> Loading testing datasets:")

# Print parameters
print ("===> HyperParameters:")
print ("	upscale_factor:		{}".format(opt.upscale_factor))
print ("	learning_rate:		{}".format(opt.lr))
print ("	batch_size:		{}".format(opt.batchSize))
print ("	nEpochs:		{}".format(opt.nEpochs))
print ("===> Best result from epoch_{}:".format(position))
print ("	min.Loss: 	{:.6f}".format(min_loss))

final_model = torch.load("{}/model_epoch_{}.pth".format(model_save_path, position))

for iteration, test_dir in enumerate(Test_dirs, 0):
	test_set = DatasetFromFolder(Test_dirs[iteration], opt.upscale_factor, False)
	testing_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

	avg_psnr_predict = 0
	avg_psnr_bicubic = 0
	for batch in testing_dataloader:
		input, target, bicubic = batch[0].to(device), batch[1], batch[2]

		predict = final_model(input)

		predict = predict.cpu()
		predict = predict.data[0].numpy().astype(np.float32)
		predict = predict * 255.
		predict[predict < 0] = 0
		predict[predict > 255.] = 255.

		target = target.data[0].numpy().astype(np.float32)
		bicubic = bicubic.data[0].numpy().astype(np.float32)

		predict = predict[0, :, :]
		target = target[0, :, :]
		bicubic = bicubic[0, :, :]

		psnr_predict = PSNR(predict, target, shave_border=opt.upscale_factor)
		psnr_bicubic = PSNR(bicubic, target, shave_border=opt.upscale_factor)

		avg_psnr_predict += psnr_predict
		avg_psnr_bicubic += psnr_bicubic

	print("	{}: 		Avg. PSNR: {:.4f} dB		Bic. PSNR:	{:.4f} dB".format(Test_sets[iteration], avg_psnr_predict / len(testing_dataloader), avg_psnr_bicubic / len(testing_dataloader)))






