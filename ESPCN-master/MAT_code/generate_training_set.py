from __future__ import print_function
import argparse

import os
from os.path import exists, join
from os import makedirs, listdir

from torchvision.transforms import Resize
from torchvision.transforms.functional import resized_crop

from PIL import Image

parser = argparse.ArgumentParser(description='Crop images for training')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
opt = parser.parse_args()
print(opt)

train_set_name = "ILSVRC2012_img_val"
img_type = ".JPEG"

train_dir = "../{}".format(train_set_name)
processed_train = "../_{}_{}".format(train_set_name, opt.upscale_factor)
if not exists(processed_train):
	print ("===> Create folder {}".format(processed_train))
	makedirs(processed_train)

def is_image_file(filename):
    return filename.endswith(img_type) and not filename.startswith(".")

def get_valid_size(image, upscale_factor):
	valid_h = image.height - (image.height % upscale_factor)
	valid_w = image.width - (image.width % upscale_factor)

	return valid_h, valid_w

def crop_valid_image(image, upscale_factor):
	valid_size = get_valid_size(image, upscale_factor)
	image = resized_crop(image, 0, 0, valid_size[0], valid_size[1], (valid_size[0], valid_size[1]), Image.BICUBIC)

	return image

def get_sub_images(image, upscale_factor):
	height, width = get_valid_size(image, upscale_factor)
	N_h = height // (17*upscale_factor)
	N_w = width // (17*upscale_factor)
	N = N_h * N_w
	#ini
	Sub_imgs = [Resize((17*upscale_factor, 17*upscale_factor))(image)]*N
	for i in range(N_h):
		for j in range(N_w):
			Sub_imgs[i*N_w + j] = resized_crop(image, i*17*upscale_factor, j*17*upscale_factor, 17*upscale_factor, 17*upscale_factor, (17*upscale_factor, 17*upscale_factor), Image.BICUBIC)
	return Sub_imgs

print ("===> Preparing training images:")

image_root = [join(train_dir, x) for x in listdir(train_dir) if is_image_file(x)]
count = 0

for iteration1, image_path in enumerate(image_root, 0):
	image = Image.open(image_path)
	image = crop_valid_image(image, opt.upscale_factor)

	Sub_imgs = get_sub_images(image, opt.upscale_factor)

	for iteration2, sub_image in enumerate(Sub_imgs, 0):
		sub_out_path = join(processed_train, "{}{}{}".format(iteration1, iteration2, img_type))
		sub_image.save(sub_out_path)
		count += 1
	print ("===> Image{} prepared! {} sub images for image{}.".format(iteration1, iteration2, iteration1))
print ("===> Preprocessing Done, total {} sub images.".format(count))