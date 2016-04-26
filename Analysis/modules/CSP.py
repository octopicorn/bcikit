__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
import json
import numpy as np
from lib.constants import *
from lib.utils import FilterCoefficients, BCIFileToEpochs
from scipy.signal import lfilter
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP as mneCSP
from lib.constants import colors
from collections import Counter
import matplotlib.pyplot as plt
import os
import thread
import time
import sys

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
            'plot': False
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

                    #if self.debug:
                        #print "."

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
                print
                if self.debug:
                    print colors.SILVER
                    print "------------------------------------------------"
                    print "Calculation threshold of ", str(self.calculation_threshold), " reached.\n Recalculating CSP spatial filters."
                    print colors.ENDC

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
        #EE = np.dot(epoch, epoch.T)
        #cov1 = np.divide(EE,np.trace(EE))

        # get standard covariance, nothing fancy
        cov2 = np.cov(epoch)
        return cov2

    def calculateCSPFilters(self):
        """
        CSPMatrix: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)

        """
        print "-------------------------"
        print "Calculate Spatial Filters:"

        """
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
        print "calculating covariances"

        class1_epochs = self.epochs[self.y == self.class1]
        class2_epochs = self.epochs[self.y == self.class2]

        cov1 = np.zeros((self.num_channels, self.num_channels))
        cov2 = np.zeros((self.num_channels, self.num_channels))

        # calculate mean covariances for each class
        for epoch in class1_epochs:
            cov1  += self.calculateEpochCovariance(epoch)
        cov1 /= self.num_channels

        for epoch in class2_epochs:
            cov2  += self.calculateEpochCovariance(epoch)
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
        all_spatial_filters = w[:, ind].T # this had the .T transpose................................................
        # pick the top n spatial filters
        self.spatial_filters = all_spatial_filters[:self.num_spatial_filters]

        if self.debug:
            print colors.MAGENTA

            # if False:
            #     print "-----------------------------------------------------------------"
            #     print "eigenvector (e) and eigenvalues (w)"
            #     print "e",e
            #     print "w",w

            print "-----------------------------------------------------------------"
            print "Spatial filters: ", all_spatial_filters[:self.num_spatial_filters].shape\
            #print all_spatial_filters[:self.num_spatial_filters]
            print colors.ENDC

        # pick the top n spatial filters
        return self.spatial_filters

    def printChart(self):
        if 1 > 2:
            fname = '/Users/odrulea/Documents/docs/gradschool/BCI/python/data/output.data'
            if os.path.isfile(fname):
                print "removed file"
                os.remove(fname)
            else:
                print "no such file"

            with open(fname, 'w') as csvfile:
                #writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

                fig, axes = plt.subplots(nrows=self.num_channels, ncols=1)
                chart_x = range(0, len(tags))

                filter_index = 0
                for spatial_filter in self.spatial_filters:

                    # make a vector to hold final result of combined (summed) data after filters applied
                    transformed_x = np.zeros(test_window.shape[1])

                    # loop through all channels
                    channel_index = 0
                    #print "****************************"
                    #print "test_window", test_window.shape
                    #print "spatial filter", filter_index, spatial_filter.shape
                    #print test_window[0]
                    for channel in test_window:
                        # print "row_num", channel_index
                        # print "spatial filter", spatial_filter[channel_index]
                        # print "channel len", len(channel), channel

                        # apply spatial filter appropriate to the channel to apply the filter to original channel data
                        # then add the result to the filtered channels summation
                        transformed_x += spatial_filter[channel_index] * channel
                        #print "transformed_x", type(transformed_x.astype(int))
                        channel_index += 1


                    # print "------------------------------------------"
                    # print colors.BOLD_CYAN
                    # print "plotting spatial filter: index",filter_index
                    # print "X",transformed_x.shape
                    # print "y",len(tags)
                    # print colors.ENDC

                    # plot the transform
                    # plt.subplot(8,1,filter_index+1)
                    # plt.plot(chart_x, transformed_x, 'r', linewidth=1)
                    # plt.fill_between(chart_x, tags*100000 , facecolor='green', alpha=0.3)

                    #writer.writerow(transformed_x.astype(float))

                    filter_index += 1

                # show the multi-chart with all transforms
                #plt.show()

                #writer.writerow(tags)

            # print "EXITING NOW"
            # thread.interrupt_main()
            # os._exit(1)
            # exit()


    def selfEvaluation(self):

        print colors.GOLD
        print "--------------------------"
        print "Self Evaluation"


        self.y = np.array(self.y)
        self.epochs = np.array(self.epochs)
        # print "y", self.y.shape

        # calc dot product of spatial filters
        self.X = np.asarray([np.dot(self.spatial_filters, epoch) for epoch in self.epochs])
        print "training X transformed by spatial filters", self.X.shape

        # compute features (mean band power)
        # compute log variance
        self.X = np.log((self.X ** 2).mean(axis=2))
        print "training X transformed by log variance of each epoch", self.X.shape

        lda = LinearDiscriminantAnalysis()
        lda = lda.fit(self.X, self.y)
        score = lda.score(self.X, self.y)

        print "LDA training score:",score
        print "-----------------------------------------------------------------"

        print "Testing Phase: feeding test data to classifier"

        # pandas is the fastest, numpy loadtxt was 10x slower
        start = time.clock()
        test_epochs, test_y = BCIFileToEpochs(
            filename=self.test_file,
            num_channels=self.num_channels,
            max_epochs=self.calculation_threshold*2,
            filter_class_labels=self.class_labels,
            epoch_size=self.epoch_size)
        end = time.clock()
        print "loaded test file in ", str(end - start),"seconds"
        print "test epochs", test_epochs.shape

        # apply IIR filters to each channel row
        test_epochs = np.apply_along_axis( self.filterChannelData, axis=1, arr=test_epochs )
        # assemble y
        test_y = np.array(test_y)
        print "test y", test_y.shape

        # assemble X
        test_epochs = np.array(test_epochs)
        # calc dot product of spatial filters
        test_X = np.asarray([np.dot(self.spatial_filters, epoch) for epoch in test_epochs])

        # compute features (mean band power)
        # compute log variance
        test_X = np.log((test_X ** 2).mean(axis=2))

        print "-----------------------------------------------------------------"
        predictions = lda.predict(test_X)
        print "Prediction Score"

        total_right = 0.
        total_wrong = 0.
        total_predictions = 0.
        for i in xrange(len(predictions)):
            #print "predicted:", predictions[i], type(predictions[i]), "/ actual:", test_y[i], type(test_y[i])
            total_predictions += 1.
            if predictions[i] == test_y[i]:
                total_right += 1.
            else:
                total_wrong += 1.

        print
        print "bandpass filter", self.bandpass_filter_range
        print "num_epochs per class", self.calculation_threshold
        print "epoch_size", self.epoch_size
        print "CSP filters:", self.num_spatial_filters

        print
        print "total right:", total_right,"total_wrong:", total_wrong,"out of", total_predictions

        print colors.BOLD_GREEN
        print
        print "percent correct:", float(total_right/total_predictions)
        print colors.ENDC

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

