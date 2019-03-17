from PIL import Image
import numpy as np
import torch
import os
from os import listdir
from os.path import join
from torch.utils import data
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms.functional import resized_crop
import cv2

from generate_training_set import is_image_file, get_valid_size, crop_valid_image

def load_image(file_path):
	image = Image.open(file_path)
	if image.mode == "L":
		y = image
	else:
		image = image.convert("YCbCr")
		y, _, _ = image.split()
	return y

class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, upscale_factor, is_training):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
		self.upscale_factor = upscale_factor
		self.is_training = is_training

	def __getitem__(self, index):
		if self.is_training:
			image = load_image(self.image_filenames[index])
			valid_image = crop_valid_image(image, self.upscale_factor)
			
			input = valid_image.copy()
			target = valid_image.copy()
			#input = Resize((17, 17), Image.BICUBIC)(input)

			input = np.array(input).astype(np.float64)
			target = np.array(target).astype(np.float64)
			input = cv2.resize(input, (17, 17), interpolation=cv2.INTER_CUBIC)
			input[input < 0] = 0
			input[input > 255.] = 255.

			input = input / 255.
			target = target / 255.

			input = torch.from_numpy(input).float().view(-1, input.shape[0], input.shape[1])
			target = torch.from_numpy(target).float().view(-1, target.shape[0], target.shape[1])

			#input = ToTensor()(input)
			#target = ToTensor()(target)

			return input, target
		else:
			image = load_image(self.image_filenames[index])
			valid_image = crop_valid_image(image, self.upscale_factor)
			valid_size = get_valid_size(valid_image, self.upscale_factor)

			input = valid_image.copy()
			target = valid_image.copy()
			#input = Resize((valid_size[0]//self.upscale_factor, valid_size[1]//self.upscale_factor), Image.BICUBIC)(input)
			#bicubic = Resize((valid_size[0], valid_size[1]), Image.BICUBIC)(input)

			input = np.array(input).astype(np.float64)
			target = np.array(target).astype(np.float64)
			input = cv2.resize(input, None, fx=1/self.upscale_factor, fy=1/self.upscale_factor, interpolation=cv2.INTER_CUBIC)
			bicubic = cv2.resize(input, None, fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)
			input[input < 0] = 0
			input[input > 255.] = 255.
			bicubic[bicubic < 0] = 0
			bicubic[bicubic > 255.] = 255.
			input = input / 255.

			input = torch.from_numpy(input).float().view(-1, input.shape[0], input.shape[1])
			target = torch.from_numpy(target).float().view(-1, target.shape[0], target.shape[1])
			bicubic = torch.from_numpy(bicubic).float().view(-1, bicubic.shape[0], bicubic.shape[1])

			#input = ToTensor()(input)
			#target = ToTensor()(target)
			#bicubic = ToTensor()(bicubic)

			return input, target, bicubic

	def __len__(self):
		return len(self.image_filenames)