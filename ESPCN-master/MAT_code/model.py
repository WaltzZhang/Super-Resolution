import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
	def __init__(self, upscale_factor):
		super(Net, self).__init__()

		self.tanh = nn.Tanh()
		# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
		self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
		self.conv3 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

		self._initialize_weights()

	def forward(self, x):
		x = self.tanh(self.conv1(x))
		x = self.tanh(self.conv2(x))
		x = self.tanh(self.conv3(x))
		x = self.pixel_shuffle(x)
		return x

	def _initialize_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('tanh'))
		init.orthogonal_(self.conv2.weight, init.calculate_gain('tanh'))
		init.orthogonal_(self.conv3.weight)
		