import os
import torch
import random
import pickle
import sampler
import warnings
import argparse
import itertools
import numpy as np
import models as ep
import scipy.io as sio

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from matplotlib import pyplot as plt

warnings.filterwarnings('always')

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def load_data(opt, folder):
	with open('../EegData_99.txt', 'rb') as data_file:
		data, speakers = pickle.load(data_file)

	label_path = '../ARL_label.mat'
	labels = sio.loadmat(label_path)['Labels'][:, 1]

	train_idx, val_idx = get_idx(folder)

	tmp_labels = labels[train_idx]
	rus = RandomUnderSampler(sampling_strategy='majority')
	train_idx_res, train_labels_res = rus.fit_resample(train_idx.reshape(-1,1), tmp_labels)
	train_idx_res = train_idx_res.squeeze()
	train_data, train_labels, train_speakers = data[train_idx_res], labels[train_idx_res], speakers[train_idx_res]
	val_data, val_labels = data[val_idx], labels[val_idx]

	train_data = torch.from_numpy(train_data.astype('float32'))
	train_labels = torch.FloatTensor(train_labels)
	train_speakers = torch.LongTensor(train_speakers)
	train_ds = TensorDataset(train_data, train_labels, train_speakers)
	train_loader = DataLoader(train_ds, batch_size=opt.batch_size, sampler=sampler.ImbalancedDatasetSampler(train_ds))

	val_data = torch.from_numpy(val_data.astype('float32'))
	val_labels = torch.FloatTensor(val_labels)
	val_ds = TensorDataset(val_data, val_labels)
	val_loader = DataLoader(val_ds, batch_size=opt.batch_size)

	return train_loader, val_loader

def get_idx(folder):
	data_per_subject = 600
	subject_num = 99

	train_idx = []
	val_idx = []

	for i in range(subject_num):
		if (i*10)//subject_num == folder:
			for j in range(i*data_per_subject,(i+1)*data_per_subject):
				val_idx.append(j)
		else:
			for j in range(i*data_per_subject,(i+1)*data_per_subject):
				train_idx.append(j)

	return np.array(train_idx), np.array(val_idx)



def test(test_loader, model, criterion, device, opt):
	model.eval()

	running_loss = 0.
	running_acc = 0.

	with torch.no_grad():
		groundtruth = []
		prediction = []

		for i, test_data in enumerate(test_loader):
			eeg_features, labels = test_data

			eeg_features = eeg_features.to(device)
			labels = labels.to(device)

			predictions = model(eeg_features)
			loss = criterion(predictions, labels.view_as(predictions))

			running_loss += loss.item()

			groundtruth.append(labels.tolist())
			predictions = torch.sigmoid(predictions)
			prediction.append(predictions.tolist())

		test_loss = running_loss / len(test_loader)
		test_acc, test_f1, test_cm = get_eval_metrics(groundtruth, prediction)

		return test_loss, test_acc, test_f1, test_cm

def get_eval_metrics(groundtruth, prediction):
	groundtruth = list(itertools.chain.from_iterable(groundtruth))
	prediction = list(itertools.chain.from_iterable(prediction))

	groundtruth = np.array(groundtruth)
	prediction = np.array(prediction)

	for i in range(prediction.shape[0]):
		prediction[i] = (prediction[i,0] > 0.5)

	acc = accuracy_score(prediction, groundtruth)
	f1 = f1_score(prediction, groundtruth)

	cm = confusion_matrix(groundtruth, prediction, normalize='true')

	return acc, f1, cm
