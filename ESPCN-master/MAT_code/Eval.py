import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio

parser = argparse.ArgumentParser(description="Pytorch Eval")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
opt = parser.parse_args()

def PSNR(valid, output, shave_border=0):
	#height, width = valid.shape[:2]
	#valid = valid[shave_border:height - shave_border, shave_border:width - shave_border]
	#output = output[shave_border:height - shave_border, shave_border:width - shave_border]
	diff = valid - output
	rmse = math.sqrt(np.mean(diff ** 2))
	if rmse == 0:
		return 100
	return 20 * math.log10(255.0 / rmse)

image_list = glob.glob("mat_{}/*.*".format(opt.dataset))

scale = 4

#model = torch.load(final_model)
#if opt.cuda:
#	model = model.cuda()
#else:
#	model = model.cpu()

avg_psnr_bicubic = 0.0
#avg_psnr_predict = 0.0
count = 0.0
for image_name in image_list:
	#if str(scale) in image_name:
	count += 1 
	print("Processing ", image_name)
	img_valid_y = sio.loadmat(image_name)['img_valid_y']
	img_bic_y = sio.loadmat(image_name)['img_bic_y']
	#img_low_y = sio.loadmat(image_name)['img_low_y']

	img_valid_y = img_valid_y.astype(float)
	img_bic_y = img_bic_y.astype(float)
	#img_low_y = img_low_y.astype(float)

	#img_low_y = ToTensor()(img_low_y).view(1, -1, img_low_y.shape[0], img_low_y.shape[1])
	#img_low_y = img_low_y / 255.0
	#img_low_y = Variable(torch.from_numpy(img_low_y).float(), requires_grad=True).view(1, -1, img_low_y.shape[0], img_low_y.shape[1])
	#if opt.cuda:
	#	img_low_y = img_low_y.cuda()

#	img_super_y = model(img_low_y)
#	img_super_y = img_super_y.cpu()
#	img_super_y = img_super_y.data[0].numpy().astype(np.float32)
#	img_super_y = img_super_y * 255.0
#	img_super_y = img_super_y[0, :, :]

#	psnr_predict = PSNR(img_valid_y, img_super_y, shave_border=scale)
	psnr_bicubic = PSNR(img_valid_y, img_bic_y, shave_border=scale)
#	avg_psnr_predict += psnr_predict
	avg_psnr_bicubic += psnr_bicubic

print("Scale=", scale)
print("Dataset=", opt.dataset)
#print("PSNR_predict=", avg_psnr_predict/count)
print("PSNR_bicubic=", avg_psnr_bicubic/count)