import heapq
import inspect
import itertools
import math
import mne
import numpy as np
import operator
import time

from bisect import bisect
from collections import Counter
from mne.decoding import CSP, FilterEstimator
from scipy.signal import butter, lfilter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels
import sklearn.utils.validation
from sklearn.utils.validation import check_X_y, check_array

class CSPEstimator(BaseEstimator, ClassifierMixin):

	def __init__(self, picks=[0], num_votes=10, bandpass_filters=[(9.0,15.0)], epoch_bounds=[(1.5,3.0)], num_spatial_filters=6, class_labels={'left':2, 'right':3}, sfreq=100.0, epoch_full_start=-0.5, epoch_full_end=3.5, consecutive='increasing',classifier_type="lda", max_channels=None):

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
		skf = StratifiedKFold(n_splits=self.tuning_csp_num_folds, shuffle=False, random_state=seed)

		kfold = skf.split(X=np.zeros(X.shape), y=y)
		# (1 is too often found to be overfitting)
		# this number is "not inclusive", meaning if you put 1, we begin with 2
		num_folds_start = 2

		# init scores
		cvscores = {}
		for i in xrange(num_folds_start,self.num_spatial_filters):
			cvscores[i] = 0

		"""
		we will now attempt to add a loop
		"""
		#self.num_spatial_filters = 6

		for i, (train, test) in enumerate(kfold):
			# calculate CSP spatial filters
			csp = CSP(n_components=self.num_spatial_filters)
			csp.fit(X[train], y[train])


			# try all filters, from the given num down to 2
			# (1 is too often found to be overfitting)
			for j in xrange(num_folds_start,self.num_spatial_filters):
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

				#if electrode_pick_scores
				if verbose:
					print "prediction score", prediction_score, "with",num_filters_to_try,"spatial filters"

		best_num_filters = max(cvscores, key=cvscores.get)
		best_score = cvscores[best_num_filters] / i+1

		if verbose:
			print "best num filters:", best_num_filters, "(average accuracy ",best_score,")"
			print "average scores per filter num:"
			for k in cvscores:
				print k,":", cvscores[k]/i+1

		return [best_num_filters, best_score]

	def fitOLD(self, X, y, classifier_type="lda", spatial_filters_num=None):

		# validate
		X, y = check_X_y(X, y, allow_nd=True)
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		# set internal vars
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y

		# top scores are stored for final ensemble
		self.ranked_classifiers = list()
		self.ranked_scores = list()
		self.ranked_scores_opts = list()
		self.ranked_transformers = list()

		##################################################
		# split X into train and test sets, so that
		# grid search can be performed on train set only
		seed = 8
		np.random.seed(seed)
		#X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X, y, test_size=0.25, random_state=seed)

		###################################################
		#
		#	GRID SEARCH
		#
		###################################################

		# optimal electrode combinations
		#electrode_picks = list(itertools.combinations(self.picks,4)) # allow groups of 4
		electrode_indexes = np.arange(len(self.picks))
		electrode_picks = list(itertools.combinations(electrode_indexes,16)) # allow groups of 4
		print "**********************"
		print len(electrode_picks)
		#exit()
		self.ranked_electrode_picks = list()

		eval_start = time.clock()

		X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X, y, test_size=0.2, random_state=seed)

		for epoch_trim in self.epoch_bounds:
			for bandpass in self.bandpass_filters:

				# separate out inputs that are tuples
				bandpass_start,bandpass_end = bandpass
				epoch_trim_start,epoch_trim_end = epoch_trim

				# bandpass filter coefficients
				b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')

				# filter and crop TRAINING SET
				X_train = self.preprocess_X(X_TRAIN, b, a, epoch_trim_start, epoch_trim_end)
				# validate
				X_train, y_train = check_X_y(X_train, y_TRAIN, allow_nd=True)
				X_train = sklearn.utils.validation.check_array(X_train, allow_nd=True)

				# filter and crop TEST SET
				X_test = self.preprocess_X(X_TEST, b, a, epoch_trim_start, epoch_trim_end)
				# validate
				X_test, y_test = check_X_y(X_test, y_TEST, allow_nd=True)
				X_test = sklearn.utils.validation.check_array(X_test, allow_nd=True)

				###########################################################################
				# self-tune CSP to find optimal number of filters to use at these settings
				if spatial_filters_num is not None:
					best_num_filters = spatial_filters_num
				else:
					[best_num_filters, best_num_filters_score] = self.self_tune(X_train, y_train)

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
				transformer = CSP(n_components=best_num_filters, reg=None)

				for electrodes in electrode_picks:
					X_train_electrodes = X_train[:,electrodes,:]
					X_test_electrodes = X_test[:,electrodes,:]

					###############################################
					###############################################
					###############################################
					transformer.fit(X_train_electrodes, y_train)

					# use these CSP spatial filters to transform train and test
					spatial_filters_train = transformer.transform(X_train_electrodes)
					spatial_filters_test = transformer.transform(X_test_electrodes)

					# put this back in as failsafe if NaN or inf starts cropping up
					# spatial_filters_train = np.nan_to_num(spatial_filters_train)
					# check_X_y(spatial_filters_train, y_train)
					# spatial_filters_test = np.nan_to_num(spatial_filters_test)
					# check_X_y(spatial_filters_test, y_test)

					if classifier_type is "lda":
						# train LDA
						classifier = LinearDiscriminantAnalysis()
					elif classifier_type is "lr":
						classifier = LogisticRegression()
					elif classifier_type is "svm":
						classifier = SVC()

					classifier.fit(spatial_filters_train, y_train)
					score = classifier.score(spatial_filters_test, y_test)

					#if score <= 1.0:
					#print "current score",score
					# print "bandpass:",bandpass_start,bandpass_end
					# print "epoch window:",epoch_trim_start,epoch_trim_end
					#print best_num_filters,"filters chosen"

					# put in ranked order Top 10 list
					idx = bisect(self.ranked_scores, score)
					self.ranked_scores.insert(idx, score)
					self.ranked_scores_opts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=best_num_filters))
					self.ranked_classifiers.insert(idx,classifier)
					self.ranked_transformers.insert(idx,transformer)

					# encode
					encoded_electrode_picks = '_'.join(map(str,electrodes))
					# decode
					# tuple(map(int,'1_2_3'.split("_")))
					#print "trying electrodes:", encoded_electrode_picks
					self.ranked_electrode_picks.insert(idx,encoded_electrode_picks)

					if len(self.ranked_scores) > self.num_votes:
						self.ranked_scores.pop(0)
					if len(self.ranked_scores_opts) > self.num_votes:
						self.ranked_scores_opts.pop(0)
					if len(self.ranked_classifiers) > self.num_votes:
						self.ranked_classifiers.pop(0)
					if len(self.ranked_transformers) > self.num_votes:
						self.ranked_transformers.pop(0)
					if len(self.ranked_electrode_picks) > self.num_votes:
						self.ranked_electrode_picks.pop(0)

				print "current score",score
				print "bandpass:",bandpass_start,bandpass_end
				print "epoch window:",epoch_trim_start,epoch_trim_end
				print "electrodes:",encoded_electrode_picks
				#print best_num_filters,"filters chosen"

				print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
				print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
				print "    T O P  ", self.num_votes, "  C L A S S I F I E R S"
				print
				#j=1
				for i in xrange(len(self.ranked_scores)):
					print i,",",round(self.ranked_scores[i],4),",",
					print self.ranked_scores_opts[i],
					print self.ranked_electrode_picks[i]


		# finish up, set the flag to indicate "fitted" state
		self.fit_ = True

		end = time.clock()
		print "CSPEstimator fit() operation took ", str(end - eval_start),"seconds"

		# Return the classifier
		return self

	def fit(self, X, y, spatial_filters_num=None):

		# validate
		X, y = check_X_y(X, y, allow_nd=True)
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		# set internal vars
		self.classes_ = unique_labels(y)
		self.X_ = X
		self.y_ = y

		# top scores are stored for final ensemble
		self.ranked_classifiers = list()
		self.ranked_scores = list()
		self.ranked_scores_opts = list()
		self.ranked_transformers = list()
		self.ranked_electrode_picks = list()

		##################################################
		# split X into train and test sets, so that
		# grid search can be performed on train set only
		seed = 1
		np.random.seed(seed)


		###################################################
		#
		#	GRID SEARCH
		#
		###################################################


		eval_start = time.clock()

		#X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=None)
		#self.grid_search(X_TRAIN,y_TRAIN, spatial_filters_num=spatial_filters_num)
		self.grid_search(X,y, spatial_filters_num=spatial_filters_num)

		# finish up, set the flag to indicate "fitted" state
		self.fit_ = True

		end = time.clock()
		print "CSPEstimator fit() operation took ", str(end - eval_start),"seconds"

		# Return the classifier
		return self

	def grid_search(self, X, y, spatial_filters_num=None):
		"""
		:param X:
		:return:
		"""

		# fix random seed for reproducibility
		seed = 1
		np.random.seed(seed)

		# define k-fold cross validation
		k = 2

		# optimal electrode combinations
		#electrode_picks = list(itertools.combinations(self.picks,4)) # allow groups of 4
		electrode_indexes = np.arange(len(self.picks))

		# allow n=max_channels electrodes
		# will pick optimal subset of electrodes if < electrode_indexes
		# example: if there are 10 electrodes, and max_channels = 4, will pick best subset of 4 electrodes from the 10 original channels
		electrode_picks = list(itertools.combinations(electrode_indexes,self.max_channels))

		for epoch_trim in self.epoch_bounds:
			for bandpass in self.bandpass_filters:

				# separate out inputs that are tuples
				bandpass_start,bandpass_end = bandpass
				epoch_trim_start,epoch_trim_end = epoch_trim

				# bandpass filter coefficients
				b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')

				# loop through cross validation folds
				skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)
				kfold = skf.split(X=np.zeros(X.shape), y=y)

				kfold_total_score = 0
				kfold_num_scores = 0

				# init electrode subset tally for this param set
				# this is used to help pick best electrode subset
				electrode_combo_scores = {}

				for i, (train, test) in enumerate(kfold):

					# filter and crop TRAINING FOLD
					X_train = self.preprocess_X(X[train], b, a, epoch_trim_start, epoch_trim_end)
					# validate
					X_train, y_train = check_X_y(X_train, y[train], allow_nd=True)
					X_train = sklearn.utils.validation.check_array(X_train, allow_nd=True)

					# filter and crop TEST FOLD
					X_test = self.preprocess_X(X[test], b, a, epoch_trim_start, epoch_trim_end)
					# validate
					X_test, y_test = check_X_y(X_test, y[test], allow_nd=True)
					X_test = sklearn.utils.validation.check_array(X_test, allow_nd=True)

					###########################################################################
					# self-tune CSP to find optimal number of filters to use at these settings
					if spatial_filters_num is not None:
						best_num_filters = spatial_filters_num
					else:
						[best_num_filters, best_num_filters_score] = self.self_tune(X_train, y_train)

					# now use this insight to really fit with optimal CSP spatial filters
					"""
					reg : float | str | None (default None)
						if not None, allow regularization for covariance estimation
						if float, shrinkage covariance is used (0 <= shrinkage <= 1).
						if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
						or Oracle Approximating Shrinkage ('oas').
					"""
					transformer = CSP(n_components=best_num_filters, reg='ledoit_wolf')


					for electrodes in electrode_picks:
						X_train_electrodes = X_train[:,electrodes,:]
						X_test_electrodes = X_test[:,electrodes,:]

						###############################################
						###############################################
						###############################################
						transformer.fit(X_train_electrodes, y_train)

						# use these CSP spatial filters to transform train and test
						spatial_filters_train = transformer.transform(X_train_electrodes)
						spatial_filters_test = transformer.transform(X_test_electrodes)

						# put this back in as failsafe if NaN or inf starts cropping up
						# spatial_filters_train = np.nan_to_num(spatial_filters_train)
						# check_X_y(spatial_filters_train, y_train)
						# spatial_filters_test = np.nan_to_num(spatial_filters_test)
						# check_X_y(spatial_filters_test, y_test)

						if self.classifier_type is "lda":
							# train LDA
							classifier = LinearDiscriminantAnalysis()
						elif self.classifier_type is "lr":
							classifier = LogisticRegression()
						elif self.classifier_type is "svm":
							classifier = SVC(probability=True)
						elif self.classifier_type is "rf":
							classifier = RandomForestClassifier()

						classifier.fit(spatial_filters_train, y_train)
						score = classifier.score(spatial_filters_test, y_test)
						kfold_total_score = kfold_total_score + score
						kfold_num_scores = kfold_num_scores + 1
						#print "[k-fold ", i, "] [electrode set:",electrodes,"] current score",score,"cumulative", kfold_total_score

						# add electrode subset to running tally for the parameter set
						# this will keep track of best performing electrode subset
						if electrodes in electrode_combo_scores:
							electrode_combo_scores[electrodes] += round(score,2)
						else:
							electrode_combo_scores[electrodes] = round(score,2)

				# calculate average score across all k folds in this iteration
				score = kfold_total_score / kfold_num_scores
				print "average score across all",kfold_num_scores,"iterations through this parameter set = ",score," (",i+1,"folds) x (",len(electrode_picks),"electrode combinations)"

				# put in ranked order Top 10 list
				idx = bisect(self.ranked_scores, score)
				self.ranked_scores.insert(idx, score)
				self.ranked_scores_opts.insert(idx, dict(bandpass=bandpass,epoch_trim=epoch_trim,filters=best_num_filters))
				self.ranked_classifiers.insert(idx,classifier)
				self.ranked_transformers.insert(idx,transformer)

				# save best electrode_subset
				# choose best electrode subset for this parameter set
				best_electrode_subset = max(electrode_combo_scores.iteritems(), key=operator.itemgetter(1))
				print "best electrode subset for this param set:", best_electrode_subset[0],"with avg score:", best_electrode_subset[1]/ k
				# encode
				encoded_electrode_picks = '_'.join(map(str,best_electrode_subset[0]))
				# decode
				# tuple(map(int,'1_2_3'.split("_")))
				#print "trying electrodes:", encoded_electrode_picks
				self.ranked_electrode_picks.insert(idx,encoded_electrode_picks)

				if len(self.ranked_scores) > self.num_votes:
					self.ranked_scores.pop(0)
				if len(self.ranked_scores_opts) > self.num_votes:
					self.ranked_scores_opts.pop(0)
				if len(self.ranked_classifiers) > self.num_votes:
					self.ranked_classifiers.pop(0)
				if len(self.ranked_transformers) > self.num_votes:
					self.ranked_transformers.pop(0)
				if len(self.ranked_electrode_picks) > self.num_votes:
					self.ranked_electrode_picks.pop(0)

				print "current score",score
				print "bandpass:",bandpass_start,bandpass_end
				print "epoch window:",epoch_trim_start,epoch_trim_end
				print "electrodes:",encoded_electrode_picks
				print best_num_filters,"filters chosen"


				print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
				print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
				print "  T O P  ", self.num_votes, "  C L A S S I F I E R S    [k:",k,"] [seed:",seed,"]"
				print
				#j=1
				for score_index in xrange(len(self.ranked_scores)):
					print score_index,",",round(self.ranked_scores[score_index],4),",",
					print self.ranked_scores_opts[score_index],
					print self.ranked_electrode_picks[score_index]



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

	def predict(self, X, y):
		"""
		:param X:
		:return:
		"""
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		if X.ndim == 2:
			trials = 1
		if X.ndim == 3:
			trials = X.shape[0]

		predictions = np.zeros((self.num_votes, trials))
		decisions = np.zeros((self.num_votes, X.shape[0]))
		predict_probas = np.zeros((self.num_votes, trials, len(self.classes_)))
		predict_log_probas = np.zeros((self.num_votes, trials, len(self.classes_)))

		for i in xrange(len(self.ranked_scores_opts)):
			# print "----------------------------------------------"
			# print "predicting with classifier",i+1
			# print self.ranked_scores_opts[i]

			bandpass_start,bandpass_end = self.ranked_scores_opts[i]['bandpass']
			epoch_trim_start,epoch_trim_end = self.ranked_scores_opts[i]['epoch_trim']
			# bandpass filter coefficients
			b, a = butter(5, np.array([bandpass_start, bandpass_end])/(self.sfreq*0.5), 'bandpass')

			# filter and crop X
			X_predict = self.preprocess_X(X,b,a,epoch_trim_start,epoch_trim_end)


			#print "predicting with electrodes (encoded):",self.ranked_electrode_picks[i]
			electrodes = tuple(map(int,self.ranked_electrode_picks[i].split("_")))
			#print "predicting with electrodes (decoded):",electrodes

			# use CSP spatial filters to transform (extract features)
			#classification_features = self.ranked_transformers[i].transform(X_predict)
			classification_features = self.ranked_transformers[i].transform(X_predict[:,electrodes,:])

			if self.classifier_type is "rf":
				decisions[i] = 0
			else:
				decisions[i] = self.ranked_classifiers[i].decision_function(classification_features)
			#print "decision: ",decisions[i]

			predictions[i] = self.ranked_classifiers[i].predict(classification_features)
			#print "predicts: ",predictions[i]

			predict_probas[i] = self.ranked_classifiers[i].predict_proba(classification_features)
			#print "predict_proba: ",predict_probas[i]

			predict_log_probas[i] = self.ranked_classifiers[i].predict_log_proba(classification_features)

		print "**********************************************"
		#print "decisions: "
		#print decisions


		#print "predicts: "
		#print predictions


		#print "classes", self.classes_
		#print "left :2, right:3"
		#print "predict class probability: "
		#print np.around(predict_probas,4)
		# print "predict class LOG probability: "
		# print np.around(predict_log_probas,4)
		#print "decisions: ",decisions
		# print
		# print "predicts: ",predictions
		# print "REAL Y:", y
		#return np.average(predictions)
		return {'predict':predictions,'predict_proba':predict_probas, 'predict_log_proba': predict_log_probas, 'decision_function':decisions}

	def score(self, X, y):
		"""
		:param X:
		:param y:
		:return:
		"""
		X = sklearn.utils.validation.check_array(X, allow_nd=True)

		scores = np.zeros(self.num_votes)
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
			print "score: ",scores[i]

		return np.average(scores)