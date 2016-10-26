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
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import covariances
from pyriemann.clustering import Potato


class CSPEstimator(BaseEstimator, ClassifierMixin):

	def __init__(self, picks=[0], num_votes=10, bandpass_filters=[(9.0,15.0)], epoch_bounds=[(1.5,3.0)], num_spatial_filters=6, class_labels={'left':2, 'right':3}, sfreq=100.0, epoch_full_start=-0.5, epoch_full_end=3.5, consecutive='increasing'):

		self.classes_ = []

		# standard way of setting local props from args
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop("self")

		for arg, val in values.items():
			setattr(self, arg, val)

		if not self.is_number(self.num_spatial_filters):
			raise TypeError("num_spatial_filters parameter must be numeric")

		# var to hold optimal number of CSP filters
		self.best_num_filters = 2
		self.best_score = 0.0
		self.tuning_csp_num_folds = 2

		# var to hold the trained classifier
		self.featureTransformer = None
		self.classifier = None

		# top scores are stored for final ensemble
		self.ranked_classifiers = list()
		self.ranked_scores = list()
		self.ranked_scores_opts = list()
		self.ranked_transformers = list()

		self.ranked_classifiers_mdm = list()
		self.ranked_scores_mdm = list()
		self.ranked_scores_opts_mdm = list()

		self.ranked_classifiers_ts = list()
		self.ranked_scores_ts = list()
		self.ranked_scores_opts_ts = list()


		# # TODO this should be moved up into init()
		# if not self.is_number(bandpass_start):
		# 	raise TypeError("bandpass_start parameter must be numeric")
		# if not self.is_number(bandpass_end):
		# 	raise TypeError("bandpass_end parameter must be numeric")
		#
		# if not self.is_number(epoch_trim_start):
		# 	raise TypeError("epoch_trim_start parameter must be numeric")
		# if not self.is_number(epoch_trim_end):
		# 	raise TypeError("epoch_trim_end parameter must be numeric")


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

	def self_tune(self, X, y, verbose=False):
		# fix random seed for reproducibility
		seed = 5
		np.random.seed(seed)

		# define k-fold cross validation test harness
		kfold = StratifiedKFold(y=y, n_folds=self.tuning_csp_num_folds, shuffle=True, random_state=seed)

		# init scores
		cvscores = {}
		for i in xrange(1,self.num_spatial_filters):
			cvscores[i+1] = 0

		for i, (train, test) in enumerate(kfold):
			# calculate CSP spatial filters
			csp = CSP(n_components=self.num_spatial_filters)
			csp.fit(X[train], y[train])

			# try all filters, from the given num down to 2
			# (1 is too often found to be overfitting)
			for j in xrange(2,self.num_spatial_filters):
				num_filters_to_try = j

				# calculate spatial filters
				csp.n_components = num_filters_to_try
				# apply CSP filters to train data
				tuning_train_LDA_features = csp.transform(X[train])
				np.nan_to_num(tuning_train_LDA_features)
				check_X_y(tuning_train_LDA_features, y[train])

				# apply CSP filters to test data
				tuning_test_LDA_features = csp.transform(X[test])
				np.nan_to_num(tuning_test_LDA_features)
				check_X_y(tuning_test_LDA_features, y[test])


				# train LDA
				lda = LinearDiscriminantAnalysis()
				prediction_score = lda.fit(tuning_train_LDA_features, y[train]).score(tuning_test_LDA_features, y[test])

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
		# validate
		X, y = check_X_y(X, y, allow_nd=True)
		X = check_array(X, allow_nd=True)

		# set internal vars
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y

		##################################################
		# split X into train and test sets, so that
		# grid search can be performed on train set only
		seed = 7
		np.random.seed(seed)
		#X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X, y, test_size=0.25, random_state=seed)

		for epoch_trim in self.epoch_bounds:
			for bandpass in self.bandpass_filters:

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

				# X_train = np.copy(X_TRAIN)
				# X_test = np.copy(X_TEST)
				# y_train = np.copy(y_TRAIN)
				# y_test = np.copy(y_TEST)

				# separate out inputs that are tuples
				bandpass_start,bandpass_end = bandpass
				epoch_trim_start,epoch_trim_end = epoch_trim



				# bandpass filter coefficients
				b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')

				# filter and crop TRAINING SET
				X_train = self.preprocess_X(X_train, b, a, epoch_trim_start, epoch_trim_end)
				# validate
				X_train, y_train = check_X_y(X_train, y_train, allow_nd=True)
				X_train = check_array(X_train, allow_nd=True)

				# filter and crop TEST SET
				X_test = self.preprocess_X(X_test, b, a, epoch_trim_start, epoch_trim_end)
				# validate
				X_test, y_test = check_X_y(X_test, y_test, allow_nd=True)
				X_test = check_array(X_test, allow_nd=True)

				###########################################################################
				# self-tune CSP to find optimal number of filters to use at these settings
				#[best_num_filters, best_num_filters_score] = self.self_tune(X_train, y_train)
				best_num_filters = 5

				# as an option, we could tune optimal CSP filter num against complete train set
				#X_tune = self.preprocess_X(X, b, a, epoch_trim_start, epoch_trim_end)
				#[best_num_filters, best_num_filters_score] = self.self_tune(X_tune, y)

				# now use this insight to really fit with optimal CSP spatial filters
				"""
				reg : float | str | None (default None)
			        if not None, allow regularization for covariance estimation
			        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
			        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
			        or Oracle Approximating Shrinkage ('oas').
				"""
				transformer = CSP(n_components=best_num_filters, reg='ledoit_wolf')
				transformer.fit(X_train, y_train)

				# use these CSP spatial filters to transform train and test
				spatial_filters_train = transformer.transform(X_train)
				spatial_filters_test = transformer.transform(X_test)

				# put this back in as failsafe if NaN or inf starts cropping up
				# spatial_filters_train = np.nan_to_num(spatial_filters_train)
				# check_X_y(spatial_filters_train, y_train)
				# spatial_filters_test = np.nan_to_num(spatial_filters_test)
				# check_X_y(spatial_filters_test, y_test)

				# train LDA
				classifier = LinearDiscriminantAnalysis()
				classifier.fit(spatial_filters_train, y_train)
				score = classifier.score(spatial_filters_test, y_test)

				#print "current score",score
				print "bandpass:",bandpass_start,bandpass_end,"epoch window:",epoch_trim_start,epoch_trim_end
				#print best_num_filters,"filters chosen"

				# put in ranked order Top 10 list
				idx = bisect(self.ranked_scores, score)
				self.ranked_scores.insert(idx, score)
				self.ranked_scores_opts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=best_num_filters))
				self.ranked_classifiers.insert(idx,classifier)
				self.ranked_transformers.insert(idx,transformer)

				if len(self.ranked_scores) > self.num_votes:
					self.ranked_scores.pop(0)
				if len(self.ranked_scores_opts) > self.num_votes:
					self.ranked_scores_opts.pop(0)
				if len(self.ranked_classifiers) > self.num_votes:
					self.ranked_classifiers.pop(0)
				if len(self.ranked_transformers) > self.num_votes:
					self.ranked_transformers.pop(0)

				"""
				Covariance computation
				"""
				# compute covariance matrices
				cov_data_train = covariances(X=X_train)
				cov_data_test = covariances(X=X_test)

				clf_mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
				clf_mdm.fit(cov_data_train, y_train)
				score_mdm = clf_mdm.score(cov_data_test, y_test)
				# print "MDM prediction score:",score_mdm
				# put in ranked order Top 10 list
				idx = bisect(self.ranked_scores_mdm, score_mdm)
				self.ranked_scores_mdm.insert(idx, score_mdm)
				self.ranked_scores_opts_mdm.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=best_num_filters))
				self.ranked_classifiers_mdm.insert(idx,clf_mdm)

				if len(self.ranked_scores_mdm) > self.num_votes:
					self.ranked_scores_mdm.pop(0)
				if len(self.ranked_scores_opts_mdm) > self.num_votes:
					self.ranked_scores_opts_mdm.pop(0)
				if len(self.ranked_classifiers_mdm) > self.num_votes:
					self.ranked_classifiers_mdm.pop(0)



				clf_ts = TSclassifier()
				clf_ts.fit(cov_data_train, y_train)
				score_ts = clf_ts.score(cov_data_test, y_test)
				# put in ranked order Top 10 list
				idx = bisect(self.ranked_scores_ts, score_ts)
				self.ranked_scores_ts.insert(idx, score_ts)
				self.ranked_scores_opts_ts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=best_num_filters))
				self.ranked_classifiers_ts.insert(idx,clf_ts)

				if len(self.ranked_scores_ts) > self.num_votes:
					self.ranked_scores_ts.pop(0)
				if len(self.ranked_scores_opts_ts) > self.num_votes:
					self.ranked_scores_opts_ts.pop(0)
				if len(self.ranked_classifiers_ts) > self.num_votes:
					self.ranked_classifiers_ts.pop(0)

				print "CSP+LDA score:",score, "Tangent space w/LR score:", score_ts


				print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
				print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
				print "    T O P  ", self.num_votes, "  C L A S S I F I E R S"
				print
				#j=1
				for i in xrange(len(self.ranked_scores)):
					print i,",",round(self.ranked_scores[i],4),",",
					print self.ranked_scores_opts[i]
				print "-------------------------------------"
				for i in xrange(len(self.ranked_scores_ts)):
					print i,",",round(self.ranked_scores_ts[i],4),",",
					print self.ranked_scores_opts_ts[i]
				print "-------------------------------------"
				for i in xrange(len(self.ranked_scores_mdm)):
					print i,",",round(self.ranked_scores_mdm[i],4),",",
					print self.ranked_scores_opts_mdm[i]

		# finish up, set the flag to indicate "fitted" state
		self.fit_ = True

		# Return the classifier
		return self

	def preprocess_X(self, X, b, a, epoch_trim_start, epoch_trim_end,verbose=False):

		if verbose:
			print "X",X.shape,"b",b,"a",a,"epoch_trim_start",epoch_trim_start,"epoch_trim_end",epoch_trim_end


		if X.ndim == 2:
			"""
			If the incoming X is just 2 dimensional, then we can assume it's a single trial.
			This would happen usually if we're preprocessing a rolling window in online mode
			in this case, we simply apply the bandpass filter, and assume cropping is handled
			by the logic that packages rolling windows.
			"""
			# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
			return np.array([lfilter(b,a,X)])

		if X.ndim == 3:
			"""
			If the incoming X is 3 dimensional, then we can assume it's a (trials,channels,times) array
			"""
			use_epoch_bounds = False if epoch_trim_start==0.0 and epoch_trim_end==0.0 else True

			if use_epoch_bounds:
				# get the max and tmin times to crop the epoch window
				tmin = math.floor(epoch_trim_start*self.sfreq)
				tmax = math.floor(epoch_trim_end*self.sfreq)
				tmin = int(tmin)
				tmax = int(tmax)
				if verbose:
					print "tmin",tmin,"tmax",tmax

				# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
				filtered_X = np.ndarray(shape=(X.shape[0],X.shape[1],abs(tmax-tmin)))
				i = 0
				for x in X:
					# apply bandpass filter to epochs
					filtered_x = lfilter(b,a,x)
					# crop epoch into new matrix
					filtered_X[i] = filtered_x[:,tmin:tmax]
					i += 1
				return filtered_X
			else:
				# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
				filtered_X = np.ndarray(shape=(X.shape[0],X.shape[1],X.shape[2]))
				i = 0
				for x in X:
					# apply bandpass filter to epochs
					filtered_X[i] = lfilter(b,a,x)
				return filtered_X

	def predict(self, X, y):
		"""
		:param X:
		:return:
		"""
		X = check_array(X, allow_nd=True)

		if X.ndim == 2:
			trials = 1
		if X.ndim == 3:
			trials = X.shape[0]

		predictions = np.zeros((self.num_votes, trials))
		#decisions = np.zeros((self.num_votes, X.shape[0]))
		predict_probas = np.zeros((self.num_votes, trials, len(self.classes_)))

		predictions_ts = np.zeros((self.num_votes, trials))
		predict_probas_ts = np.zeros((self.num_votes, trials, len(self.classes_)))

		predictions_mdm = np.zeros((self.num_votes, trials))
		predict_probas_mdm = np.zeros((self.num_votes, trials, len(self.classes_)))

		for i in xrange(self.num_votes):
			# print "----------------------------------------------"
			# print "predicting with classifier",i+1
			# print self.ranked_scores_opts[i]

			# preprocess
			bandpass_start,bandpass_end = self.ranked_scores_opts[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts[i]['epoch_trim']
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)

			# use CSP spatial filters to transform (extract features)
			classification_features = self.ranked_transformers[i].transform(X_predict)

			#decisions[i] = self.ranked_classifiers[i].decision_function(classification_features)
			#print "decision: ",decisions[i]

			predictions[i] = self.ranked_classifiers[i].predict(classification_features)
			#print "predicts: ",predictions[i]

			predict_probas[i] = self.ranked_classifiers[i].predict_proba(classification_features)
			#print "predict_proba: ",predict_probas[i]


			# Tangent space + Logistic Regression
			# preprocess
			bandpass_start,bandpass_end = self.ranked_scores_opts_ts[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts_ts[i]['epoch_trim']
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)
			# compute covariance matrices
			cov_data = covariances(X=X_predict)
			predictions_ts[i] = self.ranked_classifiers_ts[i].predict(cov_data)
			predict_probas_ts[i] = self.ranked_classifiers_ts[i].predict_proba(cov_data)


			# preprocess
			bandpass_start,bandpass_end = self.ranked_scores_opts_mdm[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts_mdm[i]['epoch_trim']
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)
			# compute covariance matrices
			cov_data = covariances(X=X_predict)
			predictions_mdm[i] = self.ranked_classifiers_mdm[i].predict(cov_data)
			predict_probas_mdm[i] = self.ranked_classifiers_mdm[i].predict_proba(cov_data)

		print "**********************************************"
		#print "decisions: "
		#print decisions
		#print "predicts: "
		#print predictions
		#print "classes", self.classes_
		print "left :2, right:3"
		print "predict class probability: "
		print np.around(predict_probas,4)
		print np.around(predict_probas_ts,4)
		print np.around(predict_probas_mdm,4)
		print "REAL Y:", y
		#return np.average(predictions)

	def score(self, X, y):
		"""
		:param X:
		:param y:
		:return:
		"""
		X = check_array(X, allow_nd=True)

		scores = np.zeros(self.num_votes)
		scores_ts = np.zeros(self.num_votes)
		scores_mdm = np.zeros(self.num_votes)

		for i in xrange(self.num_votes):
			print "----------------------------------------------"
			print "predicting with classifier",i+1
			print self.ranked_scores_opts[i]

			bandpass_start,bandpass_end = self.ranked_scores_opts[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts[i]['epoch_trim']
			# bandpass filter coefficients
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')

			# filter and crop X
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)

			# use CSP spatial filters to transform (extract features)
			classification_features = self.ranked_transformers[i].transform(X_predict)
			scores[i] = self.ranked_classifiers[i].score(classification_features, y)

			print "score CSP+LDA: ",scores[i]

			# Tangent Space stuff
			bandpass_start,bandpass_end = self.ranked_scores_opts_ts[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts_ts[i]['epoch_trim']
			# bandpass filter coefficients
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')
			# filter and crop X
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)
			print "opts:"
			print b,a,epoch_trim_start,epoch_trim_end

			# compute covariance matrices
			cov_data = covariances(X=X_predict)
			scores_ts[i] = self.ranked_classifiers_ts[i].score(cov_data, y)


			print "score TS+LR: ",scores_ts[i]



			# MDM stuff
			bandpass_start,bandpass_end = self.ranked_scores_opts_mdm[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts_mdm[i]['epoch_trim']
			# bandpass filter coefficients
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')
			# filter and crop X
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)
			print "opts:"
			print b,a,epoch_trim_start,epoch_trim_end

			# compute covariance matrices
			cov_data = covariances(X=X_predict)
			scores_mdm[i] = self.ranked_classifiers_mdm[i].score(cov_data, y)


			print "score MDM: ",scores_ts[i]

		return np.average(scores)

def getPicks(key):
	return {
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

def getChannelSubsetMotorBand8():
	"""Return Channels names."""
	return [ 25, 26, 27, 29, 30, 31, 3, 7]

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

def preprocess_X(X, b, a, epoch_trim_start, epoch_trim_end, sfreq=100.0, verbose=False):

	if verbose:
		print "X",X.shape,"b",b,"a",a,"epoch_trim_start",epoch_trim_start,"epoch_trim_end",epoch_trim_end,"sfreq",sfreq


	if X.ndim == 2:
		"""
		If the incoming X is just 2 dimensional, then we can assume it's a single trial.
		This would happen usually if we're preprocessing a rolling window in online mode
		in this case, we simply apply the bandpass filter, and assume cropping is handled
		by the logic that packages rolling windows.
		"""
		# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
		return np.array([lfilter(b,a,X)])

	if X.ndim == 3:
		"""
		If the incoming X is 3 dimensional, then we can assume it's a (trials,channels,times) array
		"""
		# get the max and tmin times to crop the epoch window
		tmin = math.floor(epoch_trim_start*sfreq)
		tmax = math.floor(epoch_trim_end*sfreq)
		tmin = int(tmin)
		tmax = int(tmax)
		if verbose:
			print "tmin",tmin,"tmax",tmax

		# prepare a matrix to hold the processed X, with dimension appropriate for the time crop
		filtered_X = np.ndarray(shape=(X.shape[0],X.shape[1],abs(tmax-tmin)))

		i = 0
		for x in X:
			# apply bandpass filter to epochs
			filtered_x = lfilter(b,a,x)
			# crop epoch into new matrix
			filtered_X[i] = filtered_x[:,tmin:tmax]
			i += 1

		return filtered_X

def X_y_from_sliding_windows(rawArray, picks, window_size=150, window_overlap=50, verbose=False):
	####################################################
	# looping over raw data in windows
	data = rawArray._data[picks]
	labels = rawArray.pick_types(stim=True)._data

	num_windows_total = (data.shape[1] / window_overlap) - (window_size/window_overlap) + 1
	X = np.zeros((num_windows_total,len(picks),window_size))
	y = np.zeros(num_windows_total)

	if verbose:
		print "RAW data",data.shape
		print "RAW labels",labels.shape
		print "num of datapoints:", data.shape[1]
		print "num of windows will be", num_windows_total
		print "X",X.shape
		print "y",y.shape

	np.set_printoptions(suppress=True)
	j = 0
	for i in xrange(0, data.shape[1]-window_size, window_overlap):
		start = i
		end = i + window_size
		window = data[:,start:end]
		class_labels = labels[:,start:end]

		X[j] = window
		y[j] = y_from_sliding_windows(class_labels, 0.5)

		if verbose:
			print "sliding window.shape",window.shape
			print class_labels
			print j,": [",i,":",i+window_size,"]"
			print "class selected for y: ",y[j]
		j = j+1

	return [X,y]

def y_from_sliding_windows(labels, threshold=0.6):
	"""
	given an array of class labels from raw data, decide which class label is represented by the group
	based on an incoming threshold setting
	:param labels:
	:param threshold:
	:return:
	"""
	#print labels.shape
	#print labels[0]
	active = labels[0][np.where( labels[0] > 0.0 )]
	active_percent = (float(len(active))/float(labels.shape[1]))
	#print "active:",len(active),"out of",labels.shape[1],"(",active_percent,"%)"
	if active_percent >= threshold:
		#return int(active[0])
		return 1
	else:
		return 0

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
	#train_fname = "data/custom/bci4/active_train/ds1g.txt"
	#test_fname = "data/custom/bci4/active_test/ds1g.txt"

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
	bandpass_filters = get_bandpass_ranges()

	# hyperparam 2
	epoch_bounds = get_window_ranges()

	# extract X,y from train data
	read_start = time.clock()

	train_raw = RawArray(train_nparray, train_info, verbose=verbose)
	[train_X,train_y] = X_y_from_sliding_windows(train_raw, picks)

	test_raw = RawArray(test_nparray, test_info, verbose=verbose)
	[test_X,test_y] = X_y_from_sliding_windows(test_raw, picks)

	read_end = time.clock()
	print "read sliding windows in ", str(read_end - read_start),"seconds"

	print "train X",train_X.shape
	print "test X",test_X.shape
		# custom grid search
	estimator = CSPEstimator(bandpass_filters=bandpass_filters,
               epoch_bounds=[(0.0,0.0)],
               num_spatial_filters=6,
               class_labels=class_labels,
               sfreq=sfreq,
               picks=picks,
               num_votes=6,
               consecutive=True)
	estimator.fit(train_X,train_y)

	#
	print "-------------------------------------------"
	score = estimator.score(test_X,test_y)
	print "-------------------------------------------"
	print "average estimator score",score
	print


	exit()





	# just a pause here to allow visual inspection of top classifiers picked by grid search
	time.sleep(15)



	# now we go into predict mode, in which we are going over the test data using sliding windows
	# this is a simulation of what would happen if we were in "online" mode with live data
	# for each window, a prediction is given by the ensemble of top classifiers
	# next to this, we see the actual labels from the real data (i.e. the y vector)
	print "-------------------------------------------"
	print "PREDICT"
	print


	exit()

if __name__ == "__main__":
	main()