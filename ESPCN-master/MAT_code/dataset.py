import torch
from torch.autograd import Variable
import torch.utils.data as data

from os import listdir
from os.path import join

import scipy.io as sio

def is_mat_file(filename):
    return filename.endswith(".mat") and not filename.startswith(".")

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor, is_training):
        super(DatasetFromFolder, self).__init__()
        if not is_training:
            self.image_paths = [join(image_dir, x) for x in listdir(image_dir) if is_mat_file(x)]
            self.image_nums = len(self.image_paths)
        else:
            self.image_nums = sio.loadmat(image_dir)['len'][0][0]
            #print (self.image_nums)
            self.image_valid_y = sio.loadmat(image_dir)['img_valid_y']
            self.image_low_y = sio.loadmat(image_dir)['img_low_y']
        self.upscale_factor = upscale_factor
        self.is_training = is_training



    def __getitem__(self, index):
        if self.is_training:
            target = self.image_valid_y[index]
            input = self.image_low_y[index]
            input = input / 255.
            target = target / 255.

            input = Variable(torch.from_numpy(input).float()).view(-1, input.shape[0], input.shape[1])
            target = Variable(torch.from_numpy(target).float()).view(-1, target.shape[0], target.shape[1])

            return input, target
        else:
            target = sio.loadmat(self.image_paths[index])['img_valid_y']
            input = sio.loadmat(self.image_paths[index])['img_low_y']
            bicubic = sio.loadmat(self.image_paths[index])['img_bic_y']
            input = input / 255.

            input = Variable(torch.from_numpy(input).float()).view(-1, input.shape[0], input.shape[1])
            target = Variable(torch.from_numpy(target).float()).view(-1, target.shape[0], target.shape[1])
            bicubic = Variable(torch.from_numpy(bicubic).float()).view(-1, bicubic.shape[0], bicubic.shape[1])

            return input, target, bicubic


    def __len__(self):
        return self.image_nums
