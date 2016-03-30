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
from mne.decoding import CSP
from lib.constants import colors

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

        # class_label_column
        self.class_label_channel = self.module_settings["class_label_channel"] if "class_label_channel" in self.module_settings else None

        # class_labels
        self.class_labels = self.module_settings["class_labels"] if "class_labels" in self.module_settings else None
        if self.class_labels:
            self.class1, self.class2 = self.class_labels

        # calculation_threshold
        self.calculation_threshold = self.module_settings["calculation_threshold"] if "calculation_threshold" in self.module_settings else 10000

        # create the notch filter coefficients (60 Hz)
        self.notch_filter_b, self.notch_filter_a = FilterCoefficients('bandstop',250.0,np.array([59.0,61.0]))
        # create the bandpass filter (7-13Hz)
        self.bandpass_filter_b, self.bandpass_filter_a = FilterCoefficients('bandpass',250.0,np.array([7.0,13.0]))

        # X and y for classification
        self.X = []
        self.y = []

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
        self.epoch_size = 250
        self.dim3 = np.zeros((0,self.num_channels,self.epoch_size))
        print self.dim3.shape

        for classLabel in self.class_labels:
            # each bin is intialized with a num_channels x 0 matrix
            self.epochs_by_class[classLabel] = []
            self.covariances_by_class[classLabel] = []

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

                    # delete class label row
                    window = np.delete(window, (self.class_label_channel), axis=0)

                    # apply IIR filters to each channel row
                    window = np.apply_along_axis( self.filterChannelData, axis=1, arr=window )

                    # append new data to its respective class bin
                    for i in xrange(5):
                        begin = i * 250
                        end = (i+1) * 250

                        nextEpoch = window[:,begin:end]
                        #print "epoch shape ", nextEpoch.shape

                        self.epochs_by_class[classLabel].append(nextEpoch) # save epoch to its specific class bin
                        self.epochs.append(nextEpoch)   # also save the epoch to cumulative, unsegregated list of epochs (will be used for cross validation)
                        self.y.append(classLabel) # retain class label for the y array (will be used for the classifier)
                        self.dim3 = np.append(self.dim3, [nextEpoch], axis=0) # save the epoch to the standard 3 dim data structure too

                    if self.debug:
                        print "------------------------------------------------"
                        print colors.CYAN
                        print "self.epochs now has ", len(self.epochs)
                        print "self.y now has ", len(self.y)
                        print "class label ("+str(classLabel)+") trials now has: ", len(self.epochs_by_class[classLabel])
                        print "the 3 dim array now has shape ", self.dim3.shape
                        print colors.ENDC


            if self.hasThresholdBeenReached():
                """
                if we have reached the minimum threshold of data point in each class, time to run CSP algorithm
                """
                if self.debug:
                    print colors.SILVER + "------------------------------------------------"
                    print "Calculation threshold of ", str(self.calculation_threshold), " reached.\n Recalculating CSP spatial filters." + colors.ENDC

                self.calculateCSP()


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
        num_trials_class1 = len(self.epochs_by_class[self.class1])
        num_trials_class2 = len(self.epochs_by_class[self.class2])
        return num_trials_class1 >= self.calculation_threshold and num_trials_class2 >= self.calculation_threshold

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

        # get standard covariance
        cov2 = np.cov(np.nan_to_num(epoch))

        return cov2
        #print "******** COVARIANCE RESULTS ************"
        #print cov1.shape
        #print cov1

    def calculateCSP(self):
        """
        CSPMatrix: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)

        """

        # segregate the covariances by class label
        for classLabel in self.class_labels:
            for epoch in self.epochs_by_class[classLabel]:
                self.covariances_by_class[classLabel].append(self.calculateEpochCovariance(epoch))

        print "number of covariance in class -1 = ", len(self.covariances_by_class[-1])
        print "number of covariance in class 1 = ", len(self.covariances_by_class[1])

        cov1 = np.mean(self.covariances_by_class[-1], axis=0)
        cov2 = np.mean(self.covariances_by_class[1], axis=0)

        print "-----------------------------------------------------------------"
        print "normalized covariance matrices for each class"
        print "cov1",cov1
        print "cov2",cov2

        e, w = eigh(cov1,cov1+cov2, turbo=True)
        print "-----------------------------------------------------------------"
        print "eigenvector (e) and eigenvalues (w)"
        print "e",e
        print "w",w

        num_vectors = len(e)

        print "-----------------------------------------------------------------"
        print "rearrange vectors to pair (first, last), (second, second last), etc"
        ind = np.empty(num_vectors, dtype=int)
        ind[::2] = np.arange(num_vectors - 1, num_vectors // 2 - 1, -1)
        ind[1::2] = np.arange(0, num_vectors // 2)
        print ind
        # reorder by new index list
        spatial_filters = w[:, ind].T

        # pick the top n filters
        S = spatial_filters[:6]
        print "-----------------------------------------------------------------"
        print "Spatial filters: ", S

        print "-----------------------------------------------------------------"
        print "feeding X and y to classifier"
        self.y = np.array(self.y)
        self.epochs = np.array(self.epochs)
        print "y", self.y.shape, "consists of", type(self.y[0])

        #
        self.X = np.asarray([np.dot(S, epoch) for epoch in self.epochs])

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

