import os
import torch
import random
import warnings
import argparse
import numpy as np
import models as ep
import train_functions as functions

from utils import EarlyStopping

warnings.filterwarnings('always')

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def main():
	parser = argparse.ArgumentParser()

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/bl', help='relative path to log')
	parser.add_argument('--verbose', type=bool, default=True, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=4, help='number of workers for data loading')
	parser.add_argument('--time_points', type=int, default=51, help='time points in EEGNet')

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=25, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=3e-5, help='initial learning rate')
	parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

	# Model
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	acc_list, f1_list, cm_list = [], [], []
	# five-fold cross validation
	for folder in range(10):
		acc, f1, cm = train_one_folder(opt, folder)
		acc_list.append(acc)
		f1_list.append(f1)
		cm_list.append(cm)

	result = np.zeros((2,2))

	for f in range(10):
		result = [[result[i][j] + cm_list[f][i][j] for j in range(len(result[0]))] for i in range(len(result))]

	print('acc:', sum(acc_list)/len(acc_list))
	print('f1:', sum(f1_list)/len(f1_list))
	print('cm:', np.divide(result, len(cm_list)))

def train_one_folder(opt, folder):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	train_loader, val_loader = functions.load_data(opt, folder)

	# Model, optimizer and loss function
	model = ep.EEGNet(opt)
	#ep.init_weights(model)
	for param in model.parameters():
		param.requires_grad = True
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
	lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

	criterion = torch.nn.BCEWithLogitsLoss()

	best_acc = 0.
	best_f1 = 0.
	best_cm = [[0,0],
			   [0,0]]

	es = EarlyStopping(patience=opt.patience)

	# Train and validate
	for epoch in range(opt.epochs_num):
		if opt.verbose:
			print('epoch: {}/{}'.format(epoch + 1, opt.epochs_num))

		train_loss, train_acc, train_f1 = train(train_loader, model, optimizer,
												criterion, device, opt)
		val_loss, val_acc, val_f1, val_cm = functions.test(	val_loader, model,
													criterion, device, opt)

		if opt.verbose:
			print(	'train_loss: {0:.5f}'.format(train_loss),
					'train_acc: {0:.3f}'.format(train_acc),
					'train_f1: {0:.3f}'.format(train_f1),
					'val_loss: {0:.5f}'.format(val_loss),
					'val_acc: {0:.3f}'.format(val_acc),
					'val_f1: {0:.3f}'.format(val_f1))

		lr_schedule.step(val_loss)

		os.makedirs(os.path.join(opt.logger_path, str(folder)), exist_ok=True)
		model_file_name = os.path.join(opt.logger_path, str(folder), 'checkpoint.pth.tar')
		state = {'epoch': epoch+1, 'eeg_classifier': model.state_dict(), 'opt': opt}
		torch.save(state, model_file_name)

		if val_acc > best_acc:
			model_file_name = os.path.join(opt.logger_path, str(folder), 'model.pth.tar')
			torch.save(state, model_file_name)

			best_acc = val_acc

		if val_f1 > best_f1:
			best_f1 = val_f1

		if val_cm[0][0] > best_cm[0][0] and val_cm[1][1] > best_cm[1][1]:
			best_cm = val_cm

		if es.step(val_loss):
			break

	return best_acc, best_f1, best_cm

def train(train_loader, model, optimizer, criterion, device, opt):
	model.train()

	running_loss = 0.
	running_acc = 0.

	groundtruth = []
	prediction = []

	for i, train_data in enumerate(train_loader):
		eeg_features, labels, _ = train_data

		eeg_features = eeg_features.to(device)
		labels = labels.to(device)

		predictions = model(eeg_features)

		loss = criterion(predictions, labels.view_as(predictions))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		groundtruth.append(labels.tolist())
		predictions = torch.sigmoid(predictions)
		prediction.append(predictions.tolist())

		if opt.verbose and i > 0 and int(len(train_loader) / 10) > 0 and i % (int(len(train_loader) / 10)) == 0:
			print('.', flush=True, end='')

	train_loss = running_loss / len(train_loader)
	train_acc, train_f1, _ = functions.get_eval_metrics(groundtruth, prediction)

	return train_loss, train_acc, train_f1

if __name__ == '__main__':
	main()
