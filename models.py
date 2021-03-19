import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.autograd import Function, Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)

class GradientReversalFunction(Function):
	"""
	Gradient Reversal Layer from:
	Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
	Forward pass is the identity function. In the backward pass,
	the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
	"""

	@staticmethod
	def forward(ctx, x, lambda_):
		ctx.lambda_ = lambda_

		return x.clone()

	@staticmethod
	def backward(ctx, grads):
		lambda_ = ctx.lambda_
		lambda_ = grads.new_tensor(lambda_)
		dx = -lambda_ * grads

		return dx, None

class GradientReversal(torch.nn.Module):
	def __init__(self, lambda_=1):
		super(GradientReversal, self).__init__()

		self.lambda_ = lambda_

	def forward(self, x):
		return GradientReversalFunction.apply(x, self.lambda_)

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class EEGNet(nn.Module):
	def __init__(self, opt):
		super(EEGNet, self).__init__()
		self.T = opt.time_points
		self.dropout_rate = opt.dropout_rate

		# Layer 1
		self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
		self.batchnorm1 = nn.BatchNorm2d(16, False)
		self.dropout1 = nn.Dropout(self.dropout_rate)

		# Layer 2
		self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
		self.conv2 = nn.Conv2d(1, 4, (2, 32), padding=0)
		self.batchnorm2 = nn.BatchNorm2d(4, False)
		self.pooling2 = nn.MaxPool2d(2, 4)
		self.dropout2 = nn.Dropout(self.dropout_rate)

		# Layer 3
		self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
		self.conv3 = nn.Conv2d(4, 4, (8, 4),padding=0)
		self.batchnorm3 = nn.BatchNorm2d(4, False)
		self.pooling3 = nn.MaxPool2d((2, 4))
		self.dropout3 = nn.Dropout(self.dropout_rate)

		# FC Layer
		# NOTE: This dimension will depend on the number of timestamps per sample in your data.
		# I have 120 timepoints.
		self.fc1 = nn.Linear(24, 1)

	def encoder(self, x):
		# Layer 1
		x = F.elu(self.conv1(x))
		x = self.batchnorm1(x)
		x = self.dropout1(x)
		x = x.permute(0, 3, 1, 2)

		# Layer 2
		x = self.padding1(x)
		x = F.elu(self.conv2(x))
		x = self.batchnorm2(x)
		x = self.pooling2(x)
		x = self.dropout2(x)

		return x

	def decoder(self, x):
		# Layer 3
		x = self.padding2(x)
		x = F.elu(self.conv3(x))
		x = self.batchnorm3(x)
		x = self.pooling3(x)
		x = self.dropout3(x)

		# FC Layer
		x = x.view(-1, 24)
		x = self.fc1(x)

		return x

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)

		return x

class SpeakerDiscriminator(nn.Module):
	def __init__(self, opt):
		super(SpeakerDiscriminator, self).__init__()

		self.subject_num = opt.subject_num
		self.linear_width = opt.linear_width

		self.flatten = Flatten()
		self.grl = GradientReversal(opt.subject_weight)

		self.linear_1 = nn.Linear(self.linear_width, self.linear_width)
		self.linear_2 = nn.Linear(self.linear_width, self.subject_num)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.flatten(x)
		x = self.grl(x)
		x = self.relu(self.linear_1(x))
		x = self.linear_2(x)

		return x
