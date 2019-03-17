import os
from os.path import join
from os import makedirs, listdir

from PIL import Image

def is_image_file(file_name):
	return any(file_name.endswith(extension) for extension in [".png", ".jpg", "jpeg", "JPEG", ".bmp"]) and not file_name.startswith(".")

def get_valid_size(image, upscale_factor):
	valid_h = image.height - (image.height % upscale_factor)
	valid_w = image.width - (image.width % upscale_factor)

	return valid_h, valid_w

def crop_valid_image(image, upscale_factor):
	valid_size = get_valid_size(image, upscale_factor)
	image = image.crop((0, 0, valid_size[1], valid_size[0]))

	return image

def get_sub_images(image, upscale_factor):
	height, width = get_valid_size(image, upscale_factor)
	N_h = height // (17*upscale_factor)
	N_w = width // (17*upscale_factor)
	N = N_h * N_w
	#ini
	sub_images = [image.resize((17*upscale_factor, 17*upscale_factor))]*N
	for i in range(N_h):
		for j in range(N_w):
			sub_images[i*N_w + j] = image.crop((j*17*upscale_factor, i*17*upscale_factor, (j+1)*17*upscale_factor, (i+1)*17*upscale_factor))
	return sub_images

def generate_training_set(train_set, train_dir, train_image_type, upscale_factor):
	print ("===>> Preparing training images:")
	makedirs(train_dir)
	origin_train_dir = "../{}".format(train_set)
	Image_paths = [join(origin_train_dir, x) for x in listdir(origin_train_dir) if is_image_file(x)]
	total_subs = 0

	for iteration1, img_path in enumerate(Image_paths, 0):
		image = Image.open(img_path)
		image = crop_valid_image(image, upscale_factor)

		Sub_images = get_sub_images(image, upscale_factor)

		for iteration2, sub_image in enumerate(Sub_images, 0):
			sub_out_path = join(train_dir, "{}_{}.{}".format(iteration1+1, iteration2+1, train_image_type))
			sub_image.save(sub_out_path)
			total_subs += 1
		print ("===>> Image {}	prepared! {}	sub-images for image{}.".format(iteration1+1, iteration2+1, iteration1+1))
	print ("===>> Preprocessing done. Total {}	sub-images.".format(total_subs))