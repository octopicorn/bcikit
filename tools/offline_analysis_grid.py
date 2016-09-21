import matplotlib as mpl
mpl.use('TkAgg')

import argparse
from bisect import bisect
import copy
import collections
import inspect
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

		self.b, self.a = butter(5, np.array([self.bandpass_start, self.bandpass_end])/(self.sfreq/2.0), 'bandpass')

		# var to hold optimal number of CSP filters
		self.best_num_filters = 2
		self.best_score = 0.0

		# var to hold the trained classifier
		self.featureTransformer = None
		self.classifier = None

		# print "Estimator initialized with hyperparameters:"
		# print "bandpass",self.bandpass_start, self.bandpass_end
		# print "epoch window",self.epoch_trim_start,self.epoch_trim_end
		# print "max CSP filter num",self.num_spatial_filters



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
		best_num = 0
		best_score = 0.0

		#[tuning_train_X, tuning_test_X, tuning_train_y, tuning_test_y] = train_test_split(X, y, test_size=0.5)
		#print "tuning_train_X", tuning_train_X.shape, "tuning_train_y", tuning_train_y.shape

		# fix random seed for reproducibility
		seed = 5
		np.random.seed(seed)

		# define 10-fold cross validation test harness
		kfold = StratifiedKFold(y=self.y_, n_folds=2, shuffle=True, random_state=seed)

		cvscores = {}
		for i in xrange(1,self.num_spatial_filters):
			cvscores[i+1] = 0

		count = 0
		for i, (train, test) in enumerate(kfold):

			# calculate CSP spatial filters
			csp = CSP(n_components=self.num_spatial_filters, reg=None, log=True)
			csp.fit(self.X_[train], self.y_[train])

			count += 1

			# try all filters, from the given num down to 2
			# (1 is too often found to be overfitting)
			for i in xrange(1,self.num_spatial_filters):
				num_filters_to_try = i+1

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
		best_score = cvscores[best_num] / count
		if verbose:
			print "best num filters:", best_num, "(average accuracy ",best_score,")"
			print "average scores per filter num:"
			for k in cvscores:
				print k,":", cvscores[k]/count

		return [best_num, best_score]

	def fit(self, X):
		"""
		incoming param rawArray contains both X and y
		"""

		if (type(X) is not mne.io.array.array.RawArray):
			raise TypeError("X must be of type mne.io.array.array.RawArray")

		# extract the X and y from the RawArray, using MNE methods
		[X, y] = self.extract_X_y(X)

		# first apply bandpass filter using scipy lfilter
		#X[self.picks] = lfilter(self.b, self.a, X[self.picks])

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

	def extract_X_y(self, rawArray, verbose=False):

		cloneRawArray = rawArray.copy()
		# # first apply bandpass filter using scipy lfilter
		cloneRawArray._data[self.picks] = lfilter(self.b, self.a, cloneRawArray._data[self.picks])

		# alternatively, we could use MNE bandpass filter
		# iir_params={'a': self.a, 'padlen': 0, 'b': self.b}

		# first pick out the events
		events = mne.find_events(cloneRawArray, shortest_event=0, consecutive=self.consecutive, verbose=verbose)

		# now use the events to pick epochs
		epochs = Epochs(raw=cloneRawArray, events=events, event_id=self.class_labels, tmin=self.epoch_full_start, tmax=self.epoch_full_end, proj=True, picks=self.picks, baseline=None, preload=True, add_eeg_ref=False, verbose=verbose)

		# trim epochs according to hyperparameter for epoch window
		epochs_trimmed = epochs.copy().crop(tmin=self.epoch_trim_start, tmax=self.epoch_trim_end)
		if verbose:
			print "train: epochs",epochs_trimmed

		X = epochs_trimmed.get_data()
		y = epochs_trimmed.events[:, -1] - 2
		if verbose:
			print "y", y.shape

		return [X, y]

	def predict(self, X):

		# extract the X and y from the RawArray, using MNE methods
		[X, y] = self.extract_X_y(X)

		sklearn.utils.validation.check_is_fitted(self, ["X_", "y_"])
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		# use CSP spatial filters to transform (extract features)
		classification_features = self.featureTransformer.transform(X)
		return self.classifier.predict(classification_features)

	def score(self, X, y=None):

		# extract the X and y from the RawArray, using MNE methods
		[X, y] = self.extract_X_y(X)

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

def analyze(train_nparray, train_info, test_nparray, test_info, verbose=False):


	#gs = GridSearchCV(estimator=CSPEstimator(), param_grid=param_grid)
	# grid_result = gs.fit(train_raw, y=[1 for i in range(len(train_raw._data[0]))])
	# print gs.best_params_
	# print grid_result
	exit()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--debug', required=False, default=False, help="debug mode (1 or 0)")
	opts = parser.parse_args()
	return opts

def print_opts(opts):
	print opts.best_num_filters,",",opts.bandpass[0],",",opts.bandpass[1],",",opts.epoch_trim_tmin,",",opts.epoch_trim_tmax

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
	possible_ranges = [(0.0,3.0), (0.5,3.0), (2.0,3.5), (1.5,3.0),  (1.0,2.0)]
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

	# variables (parameters)
	opts.bandpass = (8.0,30.0)      # bandpass filter envelope (min, max)
	opts.num_spatial_filters = 6    # max num spatial filters to try
	opts.epoch_full_tmin = -0.5     #
	opts.epoch_full_tmax = 3.5
	opts.epoch_trim_tmin = 0.0
	opts.epoch_trim_tmax = 0.0

	# create a set of many bandpass filter range combinations
	bandpass_combinations = get_bandpass_ranges()
	window_ranges = get_window_ranges()

	# vars to store cumulative performance
	best_score = 0
	best_opts = None

	# constants
	sfreq = 100.0
	opts.event_labels = {'left':2, 'right':3}

	# files
	train_fname = "data/custom/bci4/train/ds1b.txt"
	test_fname = "data/custom/bci4/test/ds1b.txt"
	#train_fname = "data/custom/bci4/active_train/ds1b.txt"
	#test_fname = "data/custom/bci4/active_test/ds1b.txt"

	# top ten scores
	ranked_scores = list()
	ranked_scores_opts = list()
	ranked_scores_lda = list()

	#################
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

	#############################################################
	print "------------------------------------------"
	print "one pass of estimator"

	# pick a subset of total electrodes, or else just get all of the channels of type 'eeg'
	picks = getPicks('motor16') or pick_types(train_info, eeg=True)

	# hyperparam 1
	bandpass_combinations = get_bandpass_ranges()

	# hyperparam 2
	epoch_bounds = get_window_ranges()

	# grid search hyperparams
	param_grid = dict(bandpass=bandpass_combinations,
	                  epoch_trim=epoch_bounds)

	# top ten scores
	ranked_scores = list()
	ranked_scores_opts = list()
	ranked_scores_lda = list()

	train_raw = RawArray(train_nparray, train_info, verbose=verbose)
	test_raw = RawArray(test_nparray, test_info, verbose=verbose)

	for epoch_trim in epoch_bounds:
		for bandpass in bandpass_combinations:
			# bandpass=(8.0,29.0),
	        #           epoch_trim=(2.0,3.5),
			foo = CSPEstimator(bandpass=bandpass,
	                   epoch_trim=epoch_trim,
	                   num_spatial_filters=6,
	                   class_labels={'left':2, 'right':3},
	                   sfreq=100.0,
	                   picks=picks,
	                   consecutive=True)
			score = foo.fit(train_raw).score(test_raw)
			print "current score",score
			#print foo.get_params()
			print "bandpass:",foo.bandpass_start,foo.bandpass_end
			print "epoch window:",foo.epoch_trim_start,foo.epoch_trim_end
			print foo.best_num_filters,"filters chosen"
			print "------------------------- done score"

			# put in ranked order Top 10 list
			idx = bisect(ranked_scores, score)
			ranked_scores.insert(idx, score)
			ranked_scores_opts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=foo.best_num_filters))

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

			print "ranked:",ranked_scores
	######################################################################################



	analyze(train_nparray, train_info, test_nparray, test_info)
	print "total run time", round(time.clock() - total_start,1),"sec"
	exit()

if __name__ == "__main__":
	main()