import matplotlib as mpl
mpl.use('TkAgg')

import argparse
from bisect import bisect
import numpy as np
import pandas as pd
import time
import itertools
import json
import pprint
import mne
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, concatenate_raws, pick_types, Epochs
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score, ShuffleSplit

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--debug', required=False, default=False, help="debug mode (1 or 0)")
	opts = parser.parse_args()
	return opts

def main():
	print "Using MNE", mne.__version__


	opts = parse_args()
	verbose = opts.debug

	# variables
	opts.bandpass = (8.0,30.0)
	opts.num_spatial_filters = 44
	opts.epoch_full_tmin = -0.5
	opts.epoch_full_tmax = 3.5
	opts.epoch_trim_tmin = 0.0
	opts.epoch_trim_tmax = 3.0


	# constants
	sfreq = 100.0
	opts.event_labels = {'left':2, 'right':3}

	# files
	train_fname = "data/custom/bci4/train/ds1g.txt"
	test_fname = "data/custom/bci4/test/ds1g.txt"

	# top ten scores
	ranked_scores = list()
	ranked_scores_opts = list()

	#################
	# get data from files
	eval_start = time.clock()
	[train_nparray, train_info] = file_to_nparray(train_fname, sfreq=sfreq, verbose=verbose)
	end = time.clock()
	print "train dataset loaded in ", str(end - eval_start),"seconds"

	eval_start = time.clock()
	[test_nparray, test_info] = file_to_nparray(test_fname, sfreq=sfreq, verbose=verbose)
	end = time.clock()
	print "test dataset loaded in ", str(end - eval_start),"seconds"

	###
	# loop through some permutations
	bandpass_combinations = []
	possible_bandpass_filter_low = np.arange(7.0, 16.0)
	possible_bandpass_filter_high = np.arange(11.0, 31.0)
	for x in itertools.product(possible_bandpass_filter_low,possible_bandpass_filter_high):
		if x[0] < x[1]:
			bandpass_combinations.append(x)

	best_score = 0
	best_opts = None

	total_start = time.clock()

	for bp in bandpass_combinations:
		print ">>*-------------------------------*>"
		print "trying with permutation", bp

		eval_start = time.clock()

		current_opts = opts
		current_opts.bandpass = bp
		#current_opts.epoch_trim_tmin = 0.0
		#current_opts.epoch_trim_tmax = 1.0

		# Bandpass filter coefficients
		# bandpass filter coefficients
		current_opts.b, current_opts.a = butter(5, np.array([current_opts.bandpass[0], current_opts.bandpass[1]])/(sfreq/2.0), 'bandpass')

		[train_X, train_y] = extract_X_and_y(train_nparray, train_info, current_opts, verbose=verbose)
		[test_X, test_y] = extract_X_and_y(test_nparray, test_info, current_opts, verbose=verbose)

		[num_trials, num_channels, num_samples] = train_X.shape

		# CLASSIFIER
		[score, best_num_filters] = eval_classification(num_channels, train_X, train_y, test_X, test_y, verbose=verbose)

		print "bandpass",current_opts.bandpass

		idx = bisect(ranked_scores, score)
		ranked_scores.insert(idx, score)
		ranked_scores_opts.insert(idx,
		                          {
			                          "filters":best_num_filters,
			                          "bandpass":current_opts.bandpass,
			                          "tmin":current_opts.epoch_trim_tmin,
			                          "tmax":current_opts.epoch_trim_tmax
		                          })

		# timer
		print round(time.clock() - eval_start,1),"sec"



		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
		print "          H A L L    O F    F A M E"
		print

		print "rank,score,filters,bandpass_low,bandpass_high,window_min,window_max"
		j=1
		for i in xrange(len(ranked_scores)-1,0,-1):
			print len(ranked_scores)-i,",",round(ranked_scores[i],4),",",ranked_scores_opts[i]['filters'],",",ranked_scores_opts[i]['bandpass'][0],",",ranked_scores_opts[i]['bandpass'][1],",",ranked_scores_opts[i]['tmin'],",",ranked_scores_opts[i]['tmax']
			j+=1
			if j>10:
				break


		if score > best_score:
			best_score = score
			best_opts = current_opts

	print "<-----&--@--------<<"
	print "best score of all permutations", best_score
	print best_opts


	print
	print
	print "total run time", round(time.clock() - total_start,1),"sec"
	print
	print



	exit()

def extract_X_and_y(raw_nparray, raw_info, opts, verbose=False):

	# need to make a new RawArray, because once we apply filter, we mutate its internal _data
	raw = RawArray(raw_nparray, raw_info, verbose=verbose)
	picks = pick_types(raw.info, eeg=True)

	# Apply band-pass filter
	raw._data[picks] = lfilter(opts.b, opts.a, raw._data[picks])

	train_events = mne.find_events(raw, shortest_event=0, verbose=verbose)
	train_epochs = Epochs(raw, train_events, opts.event_labels, tmin=opts.epoch_full_tmin, tmax=opts.epoch_full_tmax,
	                      proj=True, picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=verbose)

	epochs_trimmed = train_epochs.copy().crop(tmin=opts.epoch_trim_tmin, tmax=opts.epoch_trim_tmax)
	if verbose:
		print "train: epochs",epochs_trimmed

	X = epochs_trimmed.get_data()
	y = epochs_trimmed.events[:, -1] - 2
	if verbose:
		print "y", y.shape

	# data_full = raw._data[picks]
	# labels_full = raw._data[len(picks):]
	# if verbose:
	# 	print "labels",labels_full.shape

	return [X, y]

def eval_classification(num_spatial_filters, train_X, train_y, test_X, test_y, verbose=False):
	# Assemble a classifier

	# train / apply CSP with max num filters
	csp = CSP(n_components=num_spatial_filters, reg=None, log=True)

	csp.fit(train_X, train_y)

	best_num = 0
	best_score = 0.0

	# try at least 6 filters
	for i in xrange(6): #num_spatial_filters):

		num_filters_to_try = i+1
		if verbose:
			print "trying with first",num_filters_to_try,"spatial filters"

		# apply CSP filters to train data
		csp.n_components = num_filters_to_try
		train_feat = csp.transform(train_X)

		# apply CSP filters to test data
		test_feat = csp.transform(test_X)

		# train LDA
		lda = LinearDiscriminantAnalysis()
		prediction_score = lda.fit(train_feat, train_y).score(test_feat, test_y)

		if prediction_score > best_score:
			best_score = prediction_score
			best_num = num_filters_to_try

		if verbose:
			print "prediction score", prediction_score

	print "prediction score", best_score
	print "best filters:", best_num

	return [best_score, best_num]

def getChannelNames():
	"""Return Channels names."""
	return ['AF3', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5',
			'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'CFC7', 'CFC5', 'CFC3',
			'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz',
			'C2', 'C4', 'C6', 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4',
			'CCP6', 'CCP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5',
			'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO1', 'PO2', 'O1', 'O2']


def file_to_nparray(fname, sfreq=100.0, verbose=False):
	"""
	Create a mne raw instance from csv file.
	"""
	# get channel names
	ch_names = getChannelNames()
	ch_type = ['eeg']*len(ch_names)
	
	# add class_label as "stim" channel
	ch_names.extend(['class_label'])
	ch_type.extend(['stim'])
	
	# Read EEG file
	data = pd.read_table(fname, header=None, names=ch_names)	
	raw_data = np.array(data[ch_names]).T
	# print raw_data.shape

	# create and populate MNE info structure
	info = create_info(ch_names, sfreq=sfreq, ch_types=ch_type)
	info['filename'] = fname

	# create raw object
	return [raw_data, info]



if __name__ == "__main__":
	main()