import math
from math import log10
import numpy as np
import torch

def PSNR(source, target, shave_border):
	height, width = target.shape[:2]
	source = source[shave_border : height-shave_border, shave_border : width-shave_border]
	target = target[shave_border : height-shave_border, shave_border : width-shave_border]
	diff = source - target
	rmse = math.sqrt(np.mean(diff ** 2))
	if rmse == 0:
		return 100
	return 20 * log10(255.0 / rmse)

def Test_model(final_model, testing_dataloader, upscale_factor, device):
	avg_psnr_predict = 0
	avg_psnr_bicubic = 0
	size = len(testing_dataloader)
	for batch in testing_dataloader:
		input, target, bicubic = batch[0].to(device), batch[1], batch[2]

		predict = final_model(input)

		predict = predict.cpu()
		predict = predict.data[0].numpy().astype(np.float64)
		predict = predict * 255.
		predict[predict < 0] = 0
		predict[predict > 255.] = 255.

		target = target.data[0].numpy().astype(np.float64)
		bicubic = bicubic.data[0].numpy().astype(np.float64)

		#target = target * 255.
		#bicubic = bicubic * 255.

		predict = predict[0, :, :]
		target = target[0, :, :]
		bicubic = bicubic[0, :, :]

		psnr_predict = PSNR(predict, target, shave_border=upscale_factor)
		psnr_bicubic = PSNR(bicubic, target, shave_border=upscale_factor)

		avg_psnr_predict += psnr_predict
		avg_psnr_bicubic += psnr_bicubic

	return avg_psnr_predict/size, avg_psnr_bicubic/size