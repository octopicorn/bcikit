__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from collections import Counter
import json
from lib.constants import *
from lib.constants import colors
from lib.utils import BufferToMatrix, FilterCoefficients, BCIFileToEpochs
import matplotlib.pyplot as plt
from mne.decoding import CSP as mneCSP
import numpy as np
import os
from scipy.signal import lfilter
from scipy.linalg import eigh
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import thread
import time

"""
CSP algorithm to calculate spatial filters from incoming raw EEG

This implementation is only for a 2 class CSP
"""
class CSP(ModuleAbstract):

    MODULE_NAME = "CSP"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        if self.debug:
            print self.LOGNAME + "setup"

        self.extractSettings(self.module_settings, {
            'bandpass_filter_range': [7.0,13.0],
            'calculation_threshold': 30,
            'class_label_channel': None,
            'class_labels': None,
            'epoch_size': 250,
            'notch_filter': None,
            'num_spatial_filters': 8,
            'optimization_trace': False,
            'sampling_rate': 250,
            'separator': '\t',
            'test_file': None,
            'plot': False,
            'include_electrodes': None
        })

        # sampling_rate (Hz)
        self.sampling_rate = float(self.sampling_rate)

        # class labels
        if self.class_labels:
            self.class1, self.class2 = self.class_labels

        if self.notch_filter:
            # create the notch filter coefficients (60 Hz)
            self.notch_filter_b, self.notch_filter_a = FilterCoefficients('bandstop',self.sampling_rate,np.array(self.notch_filter))

        if self.bandpass_filter_range:
            # create the bandpass filter (7-13Hz)
            self.bandpass_filter_b, self.bandpass_filter_a = FilterCoefficients('bandpass',self.sampling_rate,np.array(self.bandpass_filter_range))

        # X and y for classification
        self.X = np.array([])
        self.y = np.array([])

        # holds epochs segregated into separate bins for each class (each dict key is a class)
        self.epochs_by_class = dict()
        # hold covariances
        self.covariances_by_class = dict()

        """
        initialize a standard 3-dim array just to have a data structure that is standard with other libs
        In most other libraries, CSP uses a 3 dimensional epoch input: (n_epochs, n_channels, n_times)
        """
        self.epochs = np.zeros((0,self.num_channels,self.epoch_size))
        self.covariances = np.zeros((0,self.num_channels,self.num_channels))
        self.all_spatial_filters = None
        self.spatial_filters = None

        # counter for epochs
        self.epochsCounter = Counter()

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == MESSAGE_TYPE_TIME_SAMPLE:
            # if the input tag is registered as one of our known inputs from conf.yml
            # use this if the input_feature is an array of json records (i.e. eeg)
            for record in buffer_content:
                if self.debug:
                    print record

        elif self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                """
                translate the base-64 encoded json in buffer to a numpy matrix object
                """
                window = BufferToMatrix(record)

                # extract class label by just getting the 0th element out of the row used for class_label
                classLabel = int(window[self.class_label_channel,0])

                # if it's one of our known class labels (and not Nan or 0), proceed with collection for CSP calculation
                if classLabel in self.class_labels:

                    # delete class label row if it exists
                    if self.class_label_channel:
                        window = np.delete(window, (self.class_label_channel), axis=0)

                    #print "incoming window of class", classLabel, window.shape

                    # apply IIR filters to each channel row
                    window = np.apply_along_axis( self.filterChannelData, axis=1, arr=window )

                    # chop incoming window into epoch_size epochs
                    window_channels, window_length = window.shape

                    start = 0
                    end = self.epoch_size
                    new_epochs = 0

                    # chop into epochs and append each epoch to running collection
                    while end <= window_length:

                        nextEpoch = np.array(window[:,start:end])

                        # save class label to the y array (will be used for the classifier)
                        self.y = np.append(self.y, classLabel)

                        # save the epoch to the standard 3 dim data structure
                        self.epochs = np.append(self.epochs, [nextEpoch], axis=0)

                        # increment counter used to tell if we've reached threshold that triggers CSP calculation
                        self.epochsCounter[classLabel] += 1

                        # next iteration
                        # this loop will run until all possible epoch_size epochs are extracted from the larger window
                        start = start + self.epoch_size
                        end = end + self.epoch_size
                        new_epochs += 1

                        sys.stdout.write(".")
                        sys.stdout.flush()

                    #if self.debug:
                        # print "finished adding", new_epochs, "epochs of size", self.epoch_size, "with", window_length-start, "time samples left out at the end"
                        # print colors.CYAN
                        # print "self.epochs now has ", len(self.epochs), " epochs total, and shape:", self.epochs.shape
                        # print "self.y now has ", len(self.y), " class labels (1 per epoch)"
                        # print len(self.epochs[[self.y==classLabel]]), "epochs for class ", str(classLabel)
                        # print colors.ENDC
                        # print "------------------------------------------------"

            if self.hasThresholdBeenReached():
                """
                if we have reached the minimum threshold of data point in each class, time to run CSP algorithm
                """
                if self.debug:
                    print "------------------------------------------------"
                    print "Calculation threshold of ", str(self.calculation_threshold), " reached"
                    print "Recalculating CSP spatial filters"

                self.calculateCSPFilters()
                self.selfEvaluation()



    def filterChannelData(self,channel_data):
        """
        apply filters to channel vector
        """
        if self.module_settings["notch_filter"]:
            # notch filter
            channel_data = lfilter(self.notch_filter_b, self.notch_filter_a, channel_data)
        # bandpass filter
        channel_data = lfilter(self.bandpass_filter_b, self.bandpass_filter_a, channel_data)
        return channel_data

    def hasThresholdBeenReached(self):
        num_trials_class1 = self.epochsCounter[self.class1]
        num_trials_class2 = self.epochsCounter[self.class2]
        #print "n1", num_trials_class1, "n2", num_trials_class2
        return (num_trials_class1 >= self.calculation_threshold) and (num_trials_class2 >= self.calculation_threshold)

    def calculateEpochCovariance(self, epoch):
        """
        calculate a normalized covariance matrix for a trial
        """
        # print "########################################"
        # remove mean from trial
        # trial = trial - trial.mean()
        # actually it turns out that cov() function already does this internally

        # get normalized covariance
        EE = np.dot(epoch, epoch.T)
        cov1 = np.divide(EE,np.trace(EE))
        return cov1

    def calculateCSPFilters(self):
        """
        CSPMatrix: the learnt CSP filters (a [channels x channels] matrix with the filters as rows)
        """

        # using the MNE CSP function to compare (reg=0.01)
        # csp = mneCSP(n_components=4, cov_est="epoch")
        # csp.fit(self.epochs, self.y)
        #
        # print colors.GREEN
        # print "CSP filters computed by MNE", csp.filters_.shape, "\n", csp.filters_
        # print colors.ENDC
        # print "-------------------------"


        # segregate the covariances by class label
        class1_epochs = self.epochs[self.y == self.class1]
        class2_epochs = self.epochs[self.y == self.class2]

        cov1 = np.zeros((self.num_channels, self.num_channels))
        cov2 = np.zeros((self.num_channels, self.num_channels))

        # calculate mean covariances for each class
        for epoch in class1_epochs:
            cov1  += np.cov(epoch)
        cov1 /= self.num_channels

        for epoch in class2_epochs:
            cov2  += np.cov(epoch)
        cov2 /= self.num_channels

        # if self.debug:
        #     print colors.YELLOW
        #     print "epochs in Class 1:", class1_epochs.shape
        #     print "mean cov class 1", cov1.shape
        #     print "epochs in Class 2:", class2_epochs.shape
        #     print "mean cov class 2", cov2.shape
        #     print colors.ENDC

        # if setting is turned on, divide by each mean covariance by trace (optional optimization step)
        if self.optimization_trace:
            # divide cov by trace for normalization
            print colors.BOLD_BLUE, "*** trace optimization step", colors.ENDC
            cov1 /= np.trace(cov1)
            cov2 /= np.trace(cov2)

        # solve the generalized eigenvector problem to find eigenvalues that maximize variance between classes
        e, w = eigh(cov1,cov1+cov2, turbo=True)

        # this subroutine simply pairs opposite numbered filter indexes
        # so for example, it would take in a list like [0 1 2 3 4 5 6 7 8]
        # and return a list like [0 8 1 7 2 6 3 5 4]
        # this allows us to take the top n pairs we want form the eigenvectors

        # number of vectors
        num_vectors = len(e)
        # rearrange vectors to pair (first, last), (second, second last), etc
        ind = np.empty(num_vectors, dtype=int)
        ind[::2] = np.arange(num_vectors - 1, num_vectors // 2 - 1, -1)
        ind[1::2] = np.arange(0, num_vectors // 2)
        # if self.debug:
        #     print "pair spatial filter indices", ind

        # reorder by new index list
        self.all_spatial_filters = w[:, ind].T # this had the .T transpose................................................
        # pick the top n spatial filters
        self.spatial_filters = self.all_spatial_filters[:self.num_spatial_filters]

        if self.debug:
            print colors.MAGENTA

            # if False:
            #     print "-----------------------------------------------------------------"
            #     print "eigenvector (e) and eigenvalues (w)"
            #     print "e",e
            #     print "w",w

            print "-----------------------------------------------------------------"
            print "Spatial filters: ", self.all_spatial_filters[:self.num_spatial_filters].shape
            #print all_spatial_filters[:self.num_spatial_filters]
            print colors.ENDC

        # pick the top n spatial filters
        return self.spatial_filters

    def printChart(self, epochs, class_labels, fname='foo.pdf', dim=3):

        print colors.RED
        print "--------------------------------"
        print "Printing chart"

        fname = '/Users/odrulea/tmp/' + fname
        print "[Save Graph to file]",fname

        if dim == 3:
            trials,channels,samples = epochs.shape
            print "trials:",trials,"channels:",channels,"samples:",samples

            chart_data = np.reshape(epochs.swapaxes(0,1),(channels,trials*samples))
            print chart_data.shape


            fig, axes = plt.subplots(nrows=channels, ncols=1)
            chart_x = np.arange(trials*samples)
            chart_class_labels = np.zeros(trials*samples)
            print class_labels
            for i in xrange(len(class_labels)):
                start = (i*samples)
                end = (i+1)*samples
                chart_class_labels[start:end] = class_labels[i]
                #print "chart_class_labels[",start,":",end,"] = ",class_labels[i]

            print "chart_data:",len(chart_data)
            #print "y", chart_class_labels.shape, chart_class_labels



            for chart_index in xrange(channels):
                # plot the transform
                plt.subplot(channels,1,chart_index+1)
                plt.plot(chart_x, chart_data[chart_index,:], 'r', linewidth=1)

                # min/max
                class_multiplier = max(abs(np.amin(chart_data[chart_index,:])), abs(np.amax(chart_data[chart_index,:])))
                #print "class_mulitplier", class_multiplier, (chart_class_labels*class_multiplier)[275:350]
                plt.fill_between(chart_x, chart_class_labels*class_multiplier , facecolor='green', alpha=0.3)

        # have to save - can't show() in multi-threaded context
        plt.savefig(fname)
        print colors.ENDC

    def printEpochs(self, epochs, class_labels, fname='foo.pdf'):

        print colors.RED
        print "--------------------------------"
        print "Printing chart"

        fname = '/Users/odrulea/tmp/' + fname
        print "[Save Graph to file]",fname

        trials,channels,samples = epochs.shape
        print "trials:",trials,"channels:",channels,"samples:",samples

        # chart_data = np.reshape(epochs.swapaxes(0,1),(channels,trials*samples))
        # print chart_data.shape
        #
        #
        # fig, axes = plt.subplots(nrows=channels, ncols=1)
        # chart_x = np.arange(trials*samples)
        # chart_class_labels = np.zeros(trials*samples)
        # print class_labels
        # for i in xrange(len(class_labels)):
        #     start = (i*samples)
        #     end = (i+1)*samples
        #     chart_class_labels[start:end] = class_labels[i]
        #     #print "chart_class_labels[",start,":",end,"] = ",class_labels[i]
        #
        # print "chart_data:",len(chart_data)
        # #print "y", chart_class_labels.shape, chart_class_labels
        #
        #
        #
        # for chart_index in xrange(channels):
        #     # plot the transform
        #     plt.subplot(channels,1,chart_index+1)
        #     plt.plot(chart_x, chart_data[chart_index,:], 'r', linewidth=1)
        #
        #     # min/max
        #     class_multiplier = max(abs(np.amin(chart_data[chart_index,:])), abs(np.amax(chart_data[chart_index,:])))
        #     #print "class_mulitplier", class_multiplier, (chart_class_labels*class_multiplier)[275:350]
        #     plt.fill_between(chart_x, chart_class_labels*class_multiplier , facecolor='green', alpha=0.3)
        #
        # # have to save - can't show() in multi-threaded context
        # plt.savefig(fname)
        # print colors.ENDC

    def extractFeatures(self, epochs, spatial_filters, chart_file_name=None, y=[]):
        print "------------------------"
        print "Extract Features"

        # calc dot product of spatial filters
        X = np.asarray([np.dot(spatial_filters, epoch) for epoch in epochs])
        print "X transformed by spatial filters", X.shape

        # CHART
        if chart_file_name and y.any():
            self.printChart(X, y)

        # compute features (mean band power)
        # compute log variance
        X = np.log((X ** 2).mean(axis=2))
        print "X transformed by log variance of each epoch", X.shape

        return X

    def tuneSpatialFilters(self):

        print colors.MAGENTA
        num_total_spatial_filters = self.all_spatial_filters.shape[0]

        best_mean = 0
        best_num = 0
        best_score = None

        for i in xrange(num_total_spatial_filters):

            num_filters_to_try = i+1
            print "trying with first",num_filters_to_try,"spatial filters"
            trial_X = self.extractFeatures(self.epochs, self.all_spatial_filters[:num_filters_to_try])
            lda = LinearDiscriminantAnalysis()
            lda = lda.fit(trial_X, self.y)
            cross_validation_folds = 10
            xval = cross_val_score(lda, trial_X, self.y, cv=cross_validation_folds)
            #print xval
            this_mean = xval.mean()
            print "mean",this_mean
            if this_mean > best_mean:
                best_mean = this_mean
                best_num = num_filters_to_try
                best_score = xval

        print "-----------------------------"
        print "best mean was", best_mean, "with", best_num, "filters used"
        print best_score

        print colors.ENDC

    def selfEvaluation(self):

        eval_start = time.clock()

        print colors.GOLD
        print "--------------------------"
        print "Self Evaluation"

        # extract features from collected epochs by transforming with spatial filters
        print "Training..."
        self.X = self.extractFeatures(self.epochs, self.spatial_filters)
        lda = LinearDiscriminantAnalysis()
        lda = lda.fit(self.X, self.y)
        cross_validation_folds = 10
        xval = cross_val_score(lda, self.X, self.y, cv=cross_validation_folds)

        self.tuneSpatialFilters()

        # print cross validation report on training LDA
        print
        print colors.BOLD_YELLOW
        print "cross-validation with k=",cross_validation_folds,"folds"
        print xval
        print "mean:", xval.mean()

        print colors.SILVER
        print "--------------------------"
        print "Self Evaluation"
        print "Testing..."

        start = time.clock()
        test_epochs, test_y = BCIFileToEpochs(
            filename=self.test_file,
            num_channels=self.num_channels,
            max_epochs_per_class=1000, #self.calculation_threshold,
            filter_class_labels=[-1,1], #self.class_labels,
            epoch_size=self.epoch_size,
            include_electrodes=self.include_electrodes
        )
        end = time.clock()
        print "loaded test file in ", str(end - start),"seconds"

        # apply IIR filters to each channel row
        test_epochs = np.apply_along_axis( self.filterChannelData, axis=1, arr=test_epochs )

        test_X = self.extractFeatures(epochs=test_epochs, spatial_filters=self.spatial_filters)
        #chart_file_name="test_filters.pdf", y=test_y)

        print "-----------------------------------------------------------------"
        print "Metrics & Score"

        print colors.ORANGE
        predicted_y = lda.predict(test_X)
        cm = confusion_matrix(test_y, predicted_y)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)


        print colors.DARK_GREEN
        print "test",self.test_file
        print "bandpass filter", self.bandpass_filter_range
        print "trained with", self.calculation_threshold, "epochs per class"
        print (self.calculation_threshold*2*self.epoch_size)/self.sampling_rate, "sec trained"
        print "epoch_size", self.epoch_size
        print "CSP filters:", self.num_spatial_filters

        print colors.BOLD_GREEN
        print "percent correct:", lda.score(test_X, test_y)
        print colors.ENDC

        end = time.clock()
        print "evaluation stage completed in ", str(end - eval_start),"seconds"

        print "########################################"
        print "########################################"
        print "########################################"
        print "########################################"

        print "EXITING NOW"
        os._exit(1)
        thread.interrupt_main()
        exit()

        exit()
        return True

