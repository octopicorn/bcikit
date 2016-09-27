import matplotlib as mpl
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
from mne.io import RawArray
from mne import create_info, concatenate_raws, pick_types, Epochs
from mne.decoding import CSP, FilterEstimator
from scipy.signal import butter, lfilter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels
import sklearn.utils.validation
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

class CSPEstimator(BaseEstimator, ClassifierMixin):

	def __init__(self, picks=[0], bandpass=(9.0,15.0), epoch_trim=(1.5,3.0), num_spatial_filters=6, class_labels={'left':2, 'right':3}, sfreq=100.0, epoch_full_start=-0.5, epoch_full_end=3.5, consecutive='increasing'):

		self.classes_ = []

		# standard way of setting local props from args
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")

		for arg, val in values.items():
			setattr(self, arg, val)

		# separate out inputs that are tuples
		self.bandpass_start,self.bandpass_end = bandpass
		self.epoch_trim_start, self.epoch_trim_end = epoch_trim

		if not self.is_number(self.bandpass_start):
			raise TypeError("bandpass_start parameter must be numeric")
		if not self.is_number(self.bandpass_end):
			raise TypeError("bandpass_end parameter must be numeric")

		if not self.is_number(self.epoch_trim_start):
			raise TypeError("epoch_trim_start parameter must be numeric")
		if not self.is_number(self.epoch_trim_end):
			raise TypeError("epoch_trim_end parameter must be numeric")

		if not self.is_number(self.num_spatial_filters):
			raise TypeError("num_spatial_filters parameter must be numeric")

		# bandpass filter coefficients
		self.b, self.a = butter(5, np.array([self.bandpass_start, self.bandpass_end])/(self.sfreq*0.5), 'bandpass')

		# var to hold optimal number of CSP filters
		self.best_num_filters = 2
		self.best_score = 0.0
		self.tuning_csp_num_folds = 2

		# var to hold the trained classifier
		self.featureTransformer = None
		self.classifier = None

	def is_number(self,s):
		try:
			float(s)
			return True
		except ValueError:
			return False

	def _check_Xy(self, X, y=None):
		"""Aux. function to check input data."""
		if y is not None:
			if len(X) != len(y) or len(y) < 1:
				raise ValueError('X and y must have the same length.')
			if X.ndim < 3:
				raise ValueError('X must have at least 3 dimensions.')

	def self_tune(self, verbose=False):
		# fix random seed for reproducibility
		seed = 5
		np.random.seed(seed)

		# define k-fold cross validation test harness
		kfold = StratifiedKFold(y=self.y_, n_folds=self.tuning_csp_num_folds, shuffle=True, random_state=seed)

		# init scores
		cvscores = {}
		for i in xrange(1,self.num_spatial_filters):
			cvscores[i+1] = 0

		for i, (train, test) in enumerate(kfold):
			# calculate CSP spatial filters
			csp = CSP(n_components=self.num_spatial_filters, reg=None, log=True)
			csp.fit(self.X_[train], self.y_[train])

			# try all filters, from the given num down to 2
			# (1 is too often found to be overfitting)
			for j in xrange(2,self.num_spatial_filters):
				num_filters_to_try = j

				# apply CSP filters to train data
				csp.n_components = num_filters_to_try
				tuning_train_LDA_features = csp.transform(self.X_[train])

				# apply CSP filters to test data
				tuning_test_LDA_features = csp.transform(self.X_[test])

				# train LDA
				lda = LinearDiscriminantAnalysis()
				prediction_score = lda.fit(tuning_train_LDA_features, self.y_[train]).score(tuning_test_LDA_features, self.y_[test])

				cvscores[num_filters_to_try] += prediction_score

				if verbose:
					print "prediction score", prediction_score, "with",num_filters_to_try,"spatial filters"

		best_num = max(cvscores, key=cvscores.get)
		best_score = cvscores[best_num] / i+1
		if verbose:
			print "best num filters:", best_num, "(average accuracy ",best_score,")"
			print "average scores per filter num:"
			for k in cvscores:
				print k,":", cvscores[k]/i+1

		return [best_num, best_score]

	def fit(self, X, y):
		# filter and crop X
		X = self.preprocess_X(X)

		# Check that X and y have correct shape
		X, y = check_X_y(X, y, allow_nd=True)

		# set internal vars
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y


		################################################
		# train / apply CSP with max num filters
		[self.best_num_filters, best_num_filters_score] = self.self_tune()

		# now use this insight to really fit
		# calculate CSP spatial filters
		csp = CSP(n_components=self.best_num_filters, reg=None, log=True)
		csp.fit(self.X_, self.y_)

		# now use CSP spatial filters to transform
		classification_features = csp.transform(self.X_)

		# train LDA
		classifier = LinearDiscriminantAnalysis()
		classifier.fit(classification_features, self.y_)

		self.featureTransformer = csp
		self.classifier = classifier

		# finish up, set the flag to indicate "fitted" state
		self.fit_ = True

		# Return the classifier
		return self

	def preprocess_X(self, X):
		# get the max and tmin times to crop the epoch window
		tmin = math.floor(self.epoch_trim_start*self.sfreq)
		tmax = math.floor(self.epoch_trim_end*self.sfreq)
		tmin = int(tmin)
		tmax = int(tmax)

		# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
		filtered_X = np.ndarray(shape=(X.shape[0],X.shape[1],abs(tmax-tmin)))
		i = 0
		for x in X:
			# apply bandpass filter to epochs
			filtered_x = lfilter(self.b,self.a,x)
			# crop epoch into new matrix
			filtered_X[i] = filtered_x[:,tmin:tmax]
			i += 1
		return filtered_X

	def predict(self, X):

		# filter and crop X
		X = self.preprocess_X(X)

		sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		# use CSP spatial filters to transform (extract features)
		classification_features = self.featureTransformer.transform(X)
		return self.classifier.predict(classification_features)

	def score(self, X, y):

		# filter and crop X
		X = self.preprocess_X(X)

		sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		# use CSP spatial filters to transform (extract features)
		classification_features = self.featureTransformer.transform(X)

		prediction_score = self.classifier.score(classification_features, y)

		return prediction_score

def getPicks(key):
	return {
        'motor16': getChannelSubsetMotorBand(),
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

def get_window_ranges():
	window_ranges = []
	possible_ranges = [(0.0,3.0),(0.5,3.0), (2.0,3.5), (1.5,3.0),  (1.0,2.0)]
	for r in possible_ranges:
		window_ranges.append(r)
	return window_ranges

def getChannelNames():
	"""Return Channels names."""
	return ['AF3', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5',
			'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'CFC7', 'CFC5', 'CFC3', 'CFC1',
			'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
			'C4', 'C6', 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6',
			'CCP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P5', 'P3',
			'P1', 'Pz', 'P2', 'P4', 'P6', 'PO1', 'PO2', 'O1', 'O2']

def getChannelSubsetMotorBand():
	"""Return Channels names."""
	return [ 24, 25, 26, 27, 28, 29,
	         30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

def getChannelSubsetFront():
	"""Return Channels names."""
	return [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def getChannelSubsetBack():
	"""Return Channels names."""
	return [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]

def file_to_nparray(fname, sfreq=100.0, verbose=False):
	"""
	Create a mne raw instance from csv file.
	"""
	# get channel names
	# in MNE, this means you must config ith two arrays:
	# 1) an array of channel names as strings
	# 2) corresponding array of channel types. in our case, all channels are type 'eeg'
	ch_names = getChannelNames()
	ch_type = ['eeg']*len(ch_names)

	# add one more channel called 'class_label' as type 'stim'
	# this type tells MNE to treat this channel of data as a class label
	ch_names.extend(['class_label'])
	ch_type.extend(['stim'])
	
	# Read EEG file
	data = pd.read_table(fname, header=None, names=ch_names)	
	raw_data = np.array(data[ch_names]).T

	# create and populate MNE info structure
	info = create_info(ch_names, sfreq=sfreq, ch_types=ch_type)
	info['filename'] = fname

	# create raw object
	return [raw_data, info]

def main():
	print "Using MNE", mne.__version__

	opts = parse_args()
	verbose = opts.debug

	# constants
	sfreq = 100.0
	class_labels = {'left':2, 'right':3}

	# files
	train_fname = "data/custom/bci4/train/ds1g.txt"
	test_fname = "data/custom/bci4/test/ds1g.txt"
	#train_fname = "data/custom/bci4/active_train/ds1b.txt"
	#test_fname = "data/custom/bci4/active_test/ds1b.txt"

	#################
	# LOAD DATA

	eval_start = time.clock()
	# load train data from training file
	[train_nparray, train_info] = file_to_nparray(train_fname, sfreq=sfreq, verbose=verbose)
	end = time.clock()
	print "train dataset", train_fname, "loaded in ", str(end - eval_start),"seconds"

	eval_start = time.clock()
	# load test data from test file
	[test_nparray, test_info] = file_to_nparray(test_fname, sfreq=sfreq, verbose=verbose)
	end = time.clock()
	print "test dataset", test_fname, "loaded in ", str(end - eval_start),"seconds"

	total_start = time.clock()

	##################
	# CLASSIFY DATA

	# pick a subset of total electrodes, or else just get all of the channels of type 'eeg'
	picks = getPicks('motor16') or pick_types(train_info, eeg=True)

	# hyperparam 1
	bandpass_combinations = get_bandpass_ranges()

	# hyperparam 2
	epoch_bounds = get_window_ranges()

	# extract X,y from train data
	train_raw = RawArray(train_nparray, train_info, verbose=verbose)
	train_events = mne.find_events(train_raw, shortest_event=0, consecutive=True, verbose=verbose)
	train_epochs = Epochs(raw=train_raw, events=train_events, event_id=class_labels, tmin=-0.5, tmax=3.5, proj=True, picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=verbose)
	train_X = train_epochs.get_data()
	train_y = train_epochs.events[:, -1] - 2

	# extract X,y from test data
	test_raw = RawArray(test_nparray, test_info, verbose=verbose)
	test_events = mne.find_events(test_raw, shortest_event=0, consecutive=True, verbose=verbose)
	test_epochs = Epochs(raw=test_raw, events=test_events, event_id=class_labels, tmin=-0.5, tmax=3.5, proj=True, picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=verbose)
	test_X = test_epochs.get_data()
	test_y = test_epochs.events[:, -1] - 2

	# # this was an attempt to use scikit-learn GridSearchCV
	# # it proves to be about 3x slower, and is also less accurate
	# seed = 7
	# np.random.seed(seed)
	# # load dataset
	# complete_X = np.concatenate((train_X, test_X), axis=0)
	# complete_y = np.concatenate((train_y, test_y), axis=0)
	#
	# # load estimator
	# estimator = CSPEstimator(class_labels=class_labels, picks=picks, consecutive=True)
	#
	# # grid search hyperparams
	# param_grid = dict(bandpass=bandpass_combinations, epoch_trim=epoch_bounds, num_spatial_filters=np.array([6]), sfreq=np.array([100.0]))
	#
	# grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
	# grid_result = grid.fit(complete_X, complete_y)
	# # summarize results
	# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	# for params, mean_score, scores in grid_result.grid_scores_:
	# 	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
	# print "total run time", round(time.clock() - total_start,1),"sec"
	# exit()

	# top ten scores
	ranked_scores = list()
	ranked_scores_opts = list()
	ranked_classifiers = list()

	# custom grid search
	for epoch_trim in epoch_bounds:
		for bandpass in bandpass_combinations:
			estimator = CSPEstimator(bandpass=bandpass,
	                   epoch_trim=epoch_trim,
	                   num_spatial_filters=6,
	                   class_labels=class_labels,
	                   sfreq=sfreq,
	                   picks=picks,
	                   consecutive=True)
			score = estimator.fit(train_X,train_y).score(test_X,test_y)
			print "current score",score
			print "bandpass:",estimator.bandpass_start,estimator.bandpass_end
			print "epoch window:",estimator.epoch_trim_start,estimator.epoch_trim_end
			print estimator.best_num_filters,"filters chosen"
			print "------------------------- done score"

			# put in ranked order Top 10 list
			idx = bisect(ranked_scores, score)
			ranked_scores.insert(idx, score)
			ranked_scores_opts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=estimator.best_num_filters))
			ranked_classifiers.insert(idx,estimator)

			if len(ranked_scores) > 10:
				ranked_scores.pop(0)
			if len(ranked_scores_opts) > 10:
				ranked_scores_opts.pop(0)
			if len(ranked_classifiers) > 10:
				ranked_classifiers.pop(0)

			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
			print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
			print "          H A L L    O F    F A M E"
			print

			print "score,filters,bandpass_low,bandpass_high,window_min,window_max"
			j=1
			for i in xrange(len(ranked_scores)-1,0,-1):
				print len(ranked_scores)-i,",",round(ranked_scores[i],4),",",
				print ranked_scores_opts[i]
				j+=1
				if j>10:
					break

			#print "ranked:",ranked_scores
	######################################################################################

	#analyze(train_nparray, train_info, test_nparray, test_info)
	print "total run time", round(time.clock() - total_start,1),"sec"
	exit()

if __name__ == "__main__":
	main()