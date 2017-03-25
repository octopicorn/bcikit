import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

import argparse
from bisect import bisect
import copy
import collections
import inspect
import math
import numpy as np
import pandas as pd
import time
import itertools
import mne
from lib.classifiers.CSPEstimator import CSPEstimator
from mne.io import RawArray
from mne import create_info, concatenate_raws, pick_types, Epochs
from mne.filter import notch_filter

from sklearn.metrics import roc_curve, auc

def getPicks(key):
	return {
		'openbci16': getChannelSubsetOpenBCI16(),
		'openbci': getChannelSubsetOpenBCI8(),
        'motor16': getChannelSubsetMotorBand(),
		'motor8': getChannelSubsetMotorBand8(),
        'front16': getChannelSubsetFront(),
		'back16': getChannelSubsetBack()
    }.get(key, None)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--debug', required=False, default=False, help="debug mode (1 or 0)")
	opts = parser.parse_args()
	return opts

def get_bandpass_ranges():
	bandpass_combinations = []
	possible_bandpass_filter_low = np.arange(8.0, 15.0)
	possible_bandpass_filter_high = np.arange(11.0, 30.0)
	for x in itertools.product(possible_bandpass_filter_low,possible_bandpass_filter_high):
		if x[0] < x[1]:
			bandpass_combinations.append(x)
	return bandpass_combinations



def getChannelNames(device="openbci16"):
	"""Return Channels names."""
	if device=="openbci16":
		return ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4',
				'channel_5', 'channel_6', 'channel_7', 'channel_8', 'channel_9',
				'channel_10', 'channel_11', 'channel_12', 'channel_13', 'channel_14',
				'channel_15']
	elif device=="openbci":
		return ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4',
				'channel_5', 'channel_6', 'channel_7']
	elif device=="bci4":
		return [
			'AF3', 'AF4', 'F5', 'F3', 'F1',
			'Fz', 'F2', 'F4', 'F6', 'FC5',
			'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
			'FC6', 'CFC7', 'CFC5', 'CFC3', 'CFC1',
			'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7',
			'C5', 'C3', 'C1', 'Cz', 'C2',
			'C4', 'C6', 'T8', 'CCP7', 'CCP5',
			'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6',
			'CCP8', 'CP5', 'CP3', 'CP1', 'CPz',
			'CP2', 'CP4', 'CP6', 'P5', 'P3',
			'P1', 'Pz', 'P2', 'P4', 'P6',
			'PO1', 'PO2', 'O1', 'O2']

def getChannelSubsetOpenBCI16():
	"""Return Channels names."""
	#return [ 0,1,2,3,8,9,12,13,14,15]
	##return [ 6,7,8,9,10,11,12,13]
	#return [ 6,7,8,9,10,11]
	return [ 8,9,10,11,12,13,14,15]
	#return [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def getChannelSubsetOpenBCI8():
	"""Return Channels names."""
	#return [ 0,1,2,3,8,9,12,13,14,15]
	##return [ 6,7,8,9,10,11,12,13]
	#return [ 6,7,8,9,10,11]
	#return [ 0,1,2,3,4,5,6,7]
	return [ 0,1,2,3,4,5,6,7]

def getChannelSubsetMotorBand():

	"""Return Channels names."""
	# return [
	# 	16, 17, 18,
	# 	21, 22, 23,
	# 	24, 25, 26,
	# 	30, 31, 32
	# ]

	# remaining TODO
	# need to figure out a sensible way to handle this electrode picking problem
	# in real life, we will have 16 electrodes
	# we could try all subsets of them in self_tune???
	return [
		#15, 16, 17, 18,
		19, 20, 21,
		25, 26, 27, 28
		#30, 31, 32, 33, 34
	]

	# return [
	# 	20, 21, 22, 24, 25, 26
	# ]


	# return [
	# 	24, 25, 26, 27,
	# #	10, 11, 13, 14,
	# 	50, 51, 52, 53,
	# #	16, 17, 18, 19
	# ]
	#return [24, 25, 26, 27, 10, 11, 13, 14]

	#return [32, 33, 34, 35, 36, 37, 38, 39]

	# return [ 24, 25, 26, 27,
	# 		 28, 29, 30, 31]

	# return [ 24, 25, 26, 27,
	# 		 28, 29, 30, 31,
	# 		 32, 33, 34, 35,
	# 		 36, 37, 38, 39]

def getChannelSubsetMotorBand8():
	"""Return Channels names."""
	return [ 19, 20, 21, 22, 25, 26, 27, 28]

def getChannelSubsetFront():
	"""Return Channels names."""
	return [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def getChannelSubsetBack():
	"""Return Channels names."""
	return [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]

def file_to_nparray(fname, device, sfreq=100.0, verbose=False, scaling_factor=None, notch=None):
	"""
	Create a mne raw instance from csv file.
	"""
	# get channel names
	# in MNE, this means you must config ith two arrays:
	# 1) an array of channel names as strings
	# 2) corresponding array of channel types. in our case, all channels are type 'eeg'
	ch_names = getChannelNames(device)
	ch_type = ['eeg']*len(ch_names)

	# add one more channel called 'class_label' as type 'stim'
	# this type tells MNE to treat this channel of data as a class label
	ch_names.extend(['class_label'])
	ch_type.extend(['stim'])

	# Read EEG file
	data = pd.read_table(fname, header=None, names=ch_names)

	# sometimes, rarely, might need to scale the data because your device recorded with an
	# incorrect order of magnitude
	if scaling_factor is not None:
		data.loc[:,:] *= scaling_factor
		data.loc[:,'class_label'] /= scaling_factor
		#print data


	raw_data = np.array(data[ch_names], dtype=np.float64).T

	if notch is not None:
		if notch == 60:
			print "Applying notch filter at 60Hz"
			# notch filter params
			notch_filter(raw_data,
						 Fs=sfreq,
						 freqs=[60],
						 filter_length=raw_data.shape[1]-1,
						 phase='zero',
						 fir_window='hamming') # 60Hzin us, 50Hz in Europe

	filtered_data = raw_data.astype(int)

	# create and populate MNE info structure
	info = create_info(ch_names, sfreq=sfreq, ch_types=ch_type)
	info['filename'] = fname

	# create raw object
	return [filtered_data, info]

def get_window_ranges():
	window_ranges = []
	#possible_ranges = [(0.0,3.0),(0.5,3.0), (2.0,3.5), (1.5,3.0)] #,  (1.0,2.0)

	# 1 sec
	#possible_ranges = [(0.0,1.0),(1.0,2.0), (2.0,3.0), (3.0,4.0), (3.5,4.5)]
	possible_ranges = [(0.0,1.0), (0.5,1.5), (1.0,2.0), (1.5,2.5), (2.0,3.0), (2.5,3.5), (3.0,4.0), (3.5,4.5)]

	# 1.5 sec
	# possible_ranges = [(0.5,2.0),(1.0,2.5), (1.5,3.0), (2.0,3.5), (2.5,4.0), (3.0,4.5)]


	# 2 sec
	# possible_ranges = [(0.5,2.5),(1.0,3.0), (1.5,3.5), (2.0,4.0), (2.5,4.5)]

	# 2.5 sec
	# possible_ranges = [(0.5,3.0),(1.0,3.5), (1.5,4.0), (2.0,4.5)]

	# 3 sec
	# possible_ranges = [(0.0,3.0), (0.5,3.5),(1.0,4.0), (1.5,4.5)]

	# 3.5 sec
	# possible_ranges = [(0.0,3.5), (0.5,4.0),(1.0,4.5)]

	# 4 sec
	# possible_ranges = [(0.0,4.0),(0.5,4.5)]

	# 4.5 sec
	# possible_ranges = [(0.0,4.5)]


	# all
	#possible_ranges = [(0.0,1.0),(0.5,1.5),	(1.0,2.0),(1.5,2.5),(2.0,3.0),	(2.5,3.5),	(3.0,4.0),	(3.5,4.5),	(0.5,2.0),	(1.0,2.5),	(1.5,3.0),	(2.0,3.5),	(2.5,4.0),	(3.0,4.5),	(0.5,2.5),(1.0,3.0), (1.5,3.5), (2.0,4.0), (2.5,4.5),(0.5,3.0),(1.0,3.5), (1.5,4.0), (2.0,4.5),(0.0,3.0), (0.5,3.5),(1.0,4.0), (1.5,4.5),(0.0,3.5), (0.5,4.0),(1.0,4.5),(0.0,4.0),(0.5,4.5),(0.0,4.5)]




	#possible_ranges = [(1.0,3.0),(2.0,4.0), (2.0,3.5), (1.5,3.0),  (1.0,2.0)]
	for r in possible_ranges:
		window_ranges.append(r)
	return window_ranges

def main():
	print "Using MNE", mne.__version__

	opts = parse_args()
	verbose = opts.debug

	num_votes = 3
	class_labels = {'left':2, 'right':3}

	draw = False

	raw_fname = None
	scaling_factor = None




	#subject = 'bci4-B'
	#subject = "bci4-G"
	# subject = 'self-A'
	# subject = 'self-D'
	#subject = 'self-E'
	subject = 'self-H'

	# constants
	if subject is 'bci4-B':
		sfreq = 100.0
		device = "bci4"
		electrode_group = 'motor16'
		train_fname = "data/custom/bci4/active_train/ds1b.txt"
		test_fname = "data/custom/bci4/active_test/ds1b.txt"
	elif subject is 'bci4-G':
		sfreq = 100.0
		device = "bci4"
		electrode_group = 'motor16'
		train_fname = "data/custom/bci4/train/ds1g.txt"
		test_fname = "data/custom/bci4/test/ds1g.txt"
	elif subject is 'self-A':
		sfreq = 125.0
		device = "openbci16"
		electrode_group = 'openbci16'
		train_fname = "data/custom/trials/motor-imagery-subject-A-train-1.csv"
		test_fname = "data/custom/trials/motor-imagery-subject-A-test-1.csv"
	elif subject is 'self-D':
		sfreq = 250.0
		device = "openbci"
		electrode_group = 'openbci'
		raw_fname = "data/custom/trials/motor-imagery-subject-D.csv"
		scaling_factor = 100.0
	elif subject is 'self-E':
		sfreq = 250.0
		device = "openbci"
		electrode_group = 'openbci'
		raw_fname = "data/custom/trials/motor-imagery-subject-E.csv"
	elif subject is 'self-H':
		sfreq = 125.0
		device = "openbci16"
		electrode_group = 'openbci16'
		raw_fname = "data/custom/trials/trial-H.csv"

	#################
	# LOAD DATA

	if raw_fname is not None:
		"""
		load train and test datasets from the same file
		"""
		eval_start = time.clock()
		# load data from file
		[raw_nparray, raw_info] = file_to_nparray(
			raw_fname,
			device,
			sfreq=sfreq,
			scaling_factor=scaling_factor,
			notch=None,
			verbose=verbose)
		end = time.clock()
		print "train & test dataset", raw_fname, "loaded in ", str(end - eval_start),"seconds"
		print raw_nparray.shape



		datalength = raw_nparray.shape[1]
		print "this session has",datalength,"datapoints"
		percent_train = 0.8
		train_length = int(round(datalength*percent_train,0))

		train_nparray = raw_nparray[:,:train_length]
		train_info = raw_info.copy()

		test_nparray = raw_nparray[:,train_length:]
		test_info = raw_info.copy()
		print "train will be",train_nparray.shape
		print "test will be",test_nparray.shape

		time.sleep(5)
		#exit()
	else:
		"""
		load train and test datasets from 2 separate files
		"""
		eval_start = time.clock()
		# load train data from training file
		[train_nparray, train_info] = file_to_nparray(train_fname, device, sfreq=sfreq, verbose=verbose)
		end = time.clock()
		print "train dataset", train_fname, "loaded in ", str(end - eval_start),"seconds"

		eval_start = time.clock()
		# load test data from test file
		[test_nparray, test_info] = file_to_nparray(test_fname, device, sfreq=sfreq, verbose=verbose)
		end = time.clock()
		print "test dataset", test_fname, "loaded in ", str(end - eval_start),"seconds"

	total_start = time.clock()

	##################
	# CLASSIFY DATA

	# pick a subset of total electrodes, or else just get all of the channels of type 'eeg'
	picks = getPicks(electrode_group) or pick_types(train_info, eeg=True)

	# hyperparam 1
	bandpass_filters = get_bandpass_ranges()

	# hyperparam 2
	epoch_bounds = get_window_ranges()

	# notch filter params
	# notches = [60, 120, 180]

	# extract X,y from train data
	print train_nparray.shape
	train_raw = RawArray(train_nparray, train_info, verbose=verbose)
	# apply notch filter
	# train_raw.notch_filter(notches)
	# filter HF above 100Hz
	# train_raw.filter(None, 100., h_trans_bandwidth=0.5, filter_length='10s', phase='zero-double')

	train_events = mne.find_events(train_raw, shortest_event=0, consecutive=True, verbose=verbose)
	train_epochs = Epochs(raw=train_raw, events=train_events, event_id=class_labels,
	                      tmin=-0.5, tmax=4.5, proj=False, picks=picks, baseline=None,
	                      preload=True, add_eeg_ref=False, verbose=verbose)
	train_X = train_epochs.get_data()
	train_y = train_epochs.events[:, -1] - 2    # convert classes [2,3] to [0,1]

	# extract X,y from test data
	test_raw = RawArray(test_nparray, test_info, verbose=verbose)
	# apply notch filter
	# test_raw.notch_filter(notches)
	# filter HF above 100Hz
	# test_raw.filter(None, 100., h_trans_bandwidth=0.5, filter_length='10s', phase='zero-double')

	test_events = mne.find_events(test_raw, shortest_event=0, consecutive=True, verbose=verbose)
	test_epochs = Epochs(raw=test_raw, events=test_events, event_id=class_labels,
	                     tmin=-0.5, tmax=4.5, proj=False, picks=picks, baseline=None,
	                     preload=True, add_eeg_ref=False, verbose=verbose)
	test_X = test_epochs.get_data()
	test_y = test_epochs.events[:, -1] - 2      # convert classes [2,3] to [0,1]



	# custom grid search
	estimator1 = CSPEstimator(
		bandpass_filters=bandpass_filters,
		epoch_bounds=epoch_bounds,
		num_spatial_filters=5,
		class_labels=class_labels,
		sfreq=sfreq,
		picks=picks,
		num_votes=num_votes,
		consecutive=True,
		classifier_type="lda",
		max_channels=len(picks))

	estimator1.fit(train_X,train_y, spatial_filters_num=None)

	estimator1_results = estimator1.predict(test_X, test_y)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	prediction_averages = []



	if draw:
		plt.figure(1)
		lw = 2
	for i in range(num_votes):
		y_score = estimator1_results['predict_log_proba'][i]
		fpr[0], tpr[0], thresholds = roc_curve(test_y, y_score[:,0], pos_label=0, drop_intermediate=False)
		roc_auc[0] = auc(fpr[0], tpr[0])

		fpr[1], tpr[1], thresholds = roc_curve(test_y, y_score[:,1], pos_label=1, drop_intermediate=False)
		roc_auc[1] = auc(fpr[1], tpr[1])

		# fpr["micro"], tpr["micro"], _ = roc_curve(test_y, y_score)
		# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		if draw:
			plt.subplot(1,num_votes,i+1)
			plt.plot(fpr[0], tpr[0], color='green',
					 lw=lw, label='AUROC: %0.2f' % roc_auc[0])
			plt.plot(fpr[1], tpr[1], color='red',
					 lw=lw, label='AUROC:: %0.2f' % roc_auc[1])
			# plt.plot(fpr["micro"], tpr["micro"], color='blue',
			# 		 lw=lw, label='AUROC: %0.2f' % roc_auc["micro"])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Classifier '+str(i+1))
			plt.legend(loc="lower right")

		print "Classifier",i+1,"ROC:",round(roc_auc[0],2),round(roc_auc[1],2)
		prediction_averages.append(roc_auc[0])

	if draw:
		plt.show()
	print "Prediction averages:", round(np.average(prediction_averages),2)
	exit()
	# ROC


	print "-------------------------------------------"
	print "-------------------------------------------"
	print "-------------------------------------------"
	print "-------------------------------------------"
	print
	time.sleep(10)



if __name__ == "__main__":
	main()