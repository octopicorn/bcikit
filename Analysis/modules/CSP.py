__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
import json
import numpy as np
from lib.constants import *
from lib.utils import FilterCoefficients
from scipy.signal import lfilter
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP as mneCSP
from lib.constants import colors
from collections import Counter
import csv
import matplotlib.pyplot as plt
import os
import thread

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

        # sampling_rate (Hz)
        self.sampling_rate = float(self.module_settings["sampling_rate"]) if "sampling_rate" in self.module_settings else 100.

        # class_label_column
        self.class_label_channel = self.module_settings["class_label_channel"] if "class_label_channel" in self.module_settings else None

        # class_labels
        self.class_labels = self.module_settings["class_labels"] if "class_labels" in self.module_settings else None
        if self.class_labels:
            self.class1, self.class2 = self.class_labels

        # number of spatial filters
        self.num_spatial_filters = 8

        # calculation_threshold
        self.calculation_threshold = self.module_settings["calculation_threshold"] if "calculation_threshold" in self.module_settings else 10000

        # bandpass filter range
        self.bandpass_filter_range = self.module_settings["bandpass_filter_range"] if "bandpass_filter_range" in self.module_settings else [7.0,13.0]
        # create the notch filter coefficients (60 Hz)
        self.notch_filter_b, self.notch_filter_a = FilterCoefficients('bandstop',self.sampling_rate,np.array([59.0,61.0]))
        # create the bandpass filter (7-13Hz)
        self.bandpass_filter_b, self.bandpass_filter_a = FilterCoefficients('bandpass',self.sampling_rate,np.array(self.bandpass_filter_range))

        # X and y for classification
        self.X = np.array([])
        self.y = np.array([])

        # holds all of the epochs from class -1 or 1, in the original order
        self.epochs = []
        # holds epochs segregated into separate bins for each class (each dict key is a class)
        self.epochs_by_class = dict()
        # hold covariances
        self.covariances_by_class = dict()

        """
        initialize a standard 3-dim array just to have a data structure that is standard with other libs
        In most other libraries, CSP uses a 3 dimensional epoch input: (n_epochs, n_channels, n_times)
        """
        self.epoch_size = self.module_settings["epoch_size"] if "epoch_size" in self.module_settings else 250
        self.epochs = np.zeros((0,self.num_channels,self.epoch_size))
        self.covariances = np.zeros((0,self.num_channels,self.num_channels))


        # counter for epochs
        self.epochsCounter = Counter()

        self.spatial_filters = None


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

                    print "incoming window of class", classLabel, window.shape

                    # apply IIR filters to each channel row
                    window = np.apply_along_axis( self.filterChannelData, axis=1, arr=window )

                    # append new data to its respective class bin

                    # chop incoming window in to epoch_size epochs
                    window_channels, window_length = window.shape

                    start = 0
                    end = self.epoch_size

                    while end <= window_length:

                        nextEpoch = np.array(window[:,start:end])

                        # save class label to the y array (will be used for the classifier)
                        self.y = np.append(self.y, classLabel)
                        print "appending ",len([nextEpoch]), " to ", self.epochs.shape

                        # save the epoch to the standard 3 dim data structure too
                        self.epochs = np.append(self.epochs, [nextEpoch], axis=0)

                        # increment counter used to tell if we've reached threshold that triggers CSP calculation
                        self.epochsCounter[classLabel] += 1

                        # next iteration
                        # this loop will run until all possible epoch_size epochs are extracted from the larger window
                        start = start + self.epoch_size
                        end = end + self.epoch_size

                    if self.debug:
                        print "------------------------------------------------"
                        print colors.CYAN
                        print "self.epochs, len:", len(self.epochs), "shape:", self.epochs.shape
                        print "self.y now has ", len(self.y)
                        print "class label ("+str(classLabel)+") epochs: ", len(self.epochs[[self.y==classLabel]])
                        print colors.ENDC


            if self.hasThresholdBeenReached():
                """
                if we have reached the minimum threshold of data point in each class, time to run CSP algorithm
                """
                if self.debug:
                    print colors.SILVER
                    print "------------------------------------------------"
                    print "Calculation threshold of ", str(self.calculation_threshold), " reached.\n Recalculating CSP spatial filters."
                    print colors.ENDC

                self.spatial_filters = self.calculateCSP()
                self.selfEvaluation()



    def filterChannelData(self,channel_data):
        """
        apply filters to channel vector
        """
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

    def calculateCSP(self):
        """
        CSPMatrix: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)

        """
        print "-------------------------"
        print "Calculate Spatial Filters:"

        """
        """

        # using the MNE CSP function to compare (reg=0.01)
        csp = mneCSP(n_components=4, cov_est="epoch")
        csp.fit(self.epochs, self.y)

        print colors.GREEN
        #print "CSP patterns ", csp.patterns_.shape, "\n", csp.patterns_
        print "CSP filters ", csp.filters_.shape, "\n", csp.filters_
        print colors.ENDC
        print "-------------------------"


        # segregate the covariances by class label
        print "calculating covariances"

        class1_epochs = self.epochs[self.y == self.class1]
        class2_epochs = self.epochs[self.y == self.class2]

        cov1 = np.zeros((self.num_channels, self.num_channels))
        cov2 = np.zeros((self.num_channels, self.num_channels))

        for epoch in class1_epochs:
            cov1  += self.calculateEpochCovariance(epoch)
        cov1 /= self.num_channels

        for epoch in class2_epochs:
            cov2  += self.calculateEpochCovariance(epoch)
        cov2 /= self.num_channels

        print colors.YELLOW
        print "mean cov class 1", cov1.shape
        print "mean cov class 2", cov2.shape
        print "epochs in Class 1:", class1_epochs.shape
        print "epochs in Class 2:", class2_epochs.shape
        print colors.ENDC

        # divide cov by trace for normalization
        cov1 /= np.trace(cov1)
        cov2 /= np.trace(cov2)

        # solve the generalized eigenvector problem to find eigenvalues that maximize variance between classes
        e, w = eigh(cov1,cov1+cov2, turbo=True)

        # number of vectors
        num_vectors = len(e)
        # rearrange vectors to pair (first, last), (second, second last), etc
        ind = np.empty(num_vectors, dtype=int)
        ind[::2] = np.arange(num_vectors - 1, num_vectors // 2 - 1, -1)
        ind[1::2] = np.arange(0, num_vectors // 2)
        if self.debug:
            print "pair spatial filter indices", ind

        # reorder by new index list
        all_spatial_filters = w[:, ind].T # this had the .T transpose................................................

        if self.debug:
            print colors.MAGENTA

            if False:
                print "-----------------------------------------------------------------"
                print "eigenvector (e) and eigenvalues (w)"
                print "e",e
                print "w",w

            print "-----------------------------------------------------------------"
            print "Spatial filters: ", all_spatial_filters[:self.num_spatial_filters]

            print colors.ENDC

        # pick the top n spatial filters
        return all_spatial_filters[:self.num_spatial_filters]

    def selfEvaluation(self):

        print "--------------------------"
        print "Self Evaluation"


        # init data window
        window = np.zeros((0,self.num_channels))
        tags = np.array([])

        # open test data file
        test_fname = "/Users/odrulea/Documents/docs/gradschool/BCI/datasets/BCI IV Competition/dataset1/train_data/BCICIV_calib_ds1a_1000Hz_cnt.txt"
        current_file = open('data/csp_oct1.data')
        csv_f = csv.reader(current_file)

        # collect all the data into window
        first_row = 0
        for row in csv_f:
            if first_row > 0:
                window = np.append(window, np.array([row[0:self.num_channels]]).astype(float), axis=0)
                tags = np.append(tags, int(row[9]))
            else:
                # skip first line (it has column headers as strings)
                first_row += 1

        print "loaded test window"

        # try charting the effect of spatial filters on bandpassed EEG signal to
        # visually inspect if they truly maximize class separation
        test_window = window.T

        # apply IIR filters to each channel row
        test_window = np.apply_along_axis( self.filterChannelData, axis=1, arr=test_window )

        # make a copy to hold the transform data, copied from the original
        test_window_transformed = np.zeros(test_window.shape)



        fname = '/Users/odrulea/Documents/docs/gradschool/BCI/python/data/output.data'
        if os.path.isfile(fname):
            print "removed file"
            os.remove(fname)
        else:
            print "no such file"

        with open(fname, 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

            fig, axes = plt.subplots(nrows=8, ncols=1)
            chart_x = range(0, len(tags))

            filter_index = 0
            for spatial_filter in self.spatial_filters:

                # make a vector to hold final result of combined (summed) data after filters applied
                transformed_x = np.zeros(test_window.shape[1])

                # loop through all channels
                channel_index = 0
                for channel in test_window:
                    #print "row_num", channel_index
                    #print "spatial filter", spatial_filter[channel_index]
                    #print "channel len", len(channel), channel

                    # apply spatial filter appropriate to the channel to apply the filter to original channel data
                    # then add the result to the filtered channels summation
                    test_window_transformed[channel_index] = spatial_filter[channel_index] * channel
                    transformed_x += spatial_filter[channel_index] * channel
                    #print "transformed_x", type(transformed_x.astype(int))
                    channel_index += 1


                print "------------------------------------------"
                print colors.BOLD_CYAN
                print "plotting spatial filter",filter_index
                print "X",transformed_x.shape
                print "y",len(tags)
                print colors.ENDC

                # plot the transform
                # plt.subplot(8,1,filter_index+1)
                # plt.plot(chart_x, transformed_x, 'r', linewidth=1)
                # plt.fill_between(chart_x, tags*100000 , facecolor='green', alpha=0.3)
                writer.writerow(transformed_x.astype(float))

                filter_index += 1

            # show the multi-chart with all transforms
            #plt.show()

            writer.writerow(tags)

        print "EXITING NOW"
        thread.interrupt_main()
        os._exit(1)
        exit()

        print "-----------------------------------------------------------------"
        print "feeding X and y to classifier"
        self.y = np.array(self.y)
        self.epochs = np.array(self.epochs)
        print "y", self.y.shape, "consists of", type(self.y[0])

        #
        self.X = np.asarray([np.dot(self.spatial_filters, epoch) for epoch in self.epochs])

        # compute features (mean band power)
        # compute log variance
        self.X = np.log((self.X ** 2).mean(axis=-1))

        print "X",self.X.shape, "consists of", type(self.X[0])

        model = LinearDiscriminantAnalysis()
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)

        print "-----------------------------------------------------------------"
        print "LDA score:",score

        print "########################################"
        print "########################################"
        print "########################################"
        print "########################################"

        exit()
        return True

