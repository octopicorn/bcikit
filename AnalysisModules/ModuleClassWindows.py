__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix, MatrixToBuffer
import json
import bisect
import itertools
import numpy as np

"""
The ClassWindows Module is meant for bounding raw EEG data into epochs by class label.

This module performs the kind of windowing used in training phase.  In this scenario, the goal is to collect windows
in which every datum belongs to the same class label.  This depends upon two inputs:

inputs:
    data:
        name: "eeg2"
        message_type: "TIME_SAMPLE"
        data_type: "EEG_RAW"
    labels:
        name: "motor_class"
        message_type: "TIME_SAMPLE"
        data_type: "CLASS_LABELS"

These must be named "data" and "labels" (deosn't matter what order), and they should both be of message_type JSON.

One key consideration when merging two input streams is that each is streaming from an asynchronous thread.  They are
almost certainly guaranteed to be out of synch with each other, especially when you take into account different output
frequency and different output buffers.  However, since both of these streams include timestamps on data, and the
timestamps are relative to a universal reference (system clock), we have an opportunity to resynchronize the streams
as we merge.

The way it works is like this: as raw EEG comes in, we simply collect it in a growing matrix.  When a class label
arrives, if it the same as the last class label, we just keep collecting.  If it is different, this is a signal that
it's time to cut off the current epoch and begin the next one, since epochs are grouped by class label.

To do this, we look at the timestamp of the new incoming class label, to see where the cutoff point will be on the
window we've collected so far.  Then, using that time marker, we slice the old data, label with the old class, and
deliver it.  The remainder becomes the new accumulator window, to be used with the new class label.
"""
class ModuleClassWindows(ModuleAbstract):

    MODULE_NAME = "Class Windows Module"

    # LOGNAME is a prefix used to prepend to debugging output statements, helps to disambiguate messages since the
    # modules run on separate threads
    LOGNAME = "[Analysis Service: Class Windows Module] "

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        self.countTotal = 0

        # Deprecated (leaving this in just in case we decide someday that we need to support
        # multiple EEG inputs with different numbers of channels)
        ## override num_channels if our input is coming from a signal generator, which could have any number of channels
        ##if 'num_channels' in self.module_conf['inputs']['data']:
        ##    self.num_channels = self.module_conf['inputs']['data']['num_channels']
        ##self.headers = ['timestamp'] + ['channel_%s' % i for i in xrange(self.num_channels)]

        self.samples_per_window = 500

        self.window = np.matrix(np.zeros((self.num_channels,0)))
        self.nextWindowSegment = np.matrix(np.zeros((self.num_channels,0)))
        self.timestamps = []
        self.lastClassLabel = 0
        self.currentClassLabel = 0
        self.currentClassLabelTimepointIndex = 0
        self.classLabels = []


        if self.debug:
            print self.LOGNAME + "setup"

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """



        # if the input tag is registered as one of our known inputs from conf.yml
        if method.consumer_tag in self.inputs.keys():
            messageType = self.inputs[method.consumer_tag]['message_type']
            if messageType == self.MESSAGE_TYPE_TIME_SAMPLE:
                # use this if the input_feature is an array of json records (i.e. eeg)
                buffer_content = json.loads(body)

                if method.consumer_tag == "data":
                    """
                    Handle incoming stream of data
                    """
                    # as long as raw data is coming in, just keep accumulating it
                    self.nextWindowSegment = np.matrix(np.zeros((self.num_channels,len(buffer_content))))
                    sample = 0
                    for record in buffer_content:
                        # get the next data out of the buffer as an array indexed by column names
                        arr = np.array([record.get(column_name, None) for column_name in self.headers])
                        self.nextWindowSegment[:,sample] = arr[1:len(self.headers)][np.newaxis].T
                        self.timestamps.append(record['timestamp'])
                        sample += 1

                    # append new segment to rolling window
                    self.window = np.hstack((self.window,self.nextWindowSegment))

                    # if self.debug:
                    #     print "----------------------------------------"
                    #     print "accumulating data"



                elif method.consumer_tag == "labels":
                    """
                    Handle incoming stream of class labels
                    """
                    for record in buffer_content:
                        # new label has come in
                        self.lastClassLabel = int(self.currentClassLabel)
                        self.currentClassLabel = int(record['class'])
                        if self.debug:
                            print "---------------------------------------------"
                            print "incoming class: " + str(record['class'])

                        if self.currentClassLabel == self.lastClassLabel:
                            """
                            new label same as last, keep accumulating
                            """

                        elif len(self.timestamps):
                            """
                            new label is different than last, time to cut a new window and deliver it
                            """

                            # 1. find out where the current label belongs
                            self.currentClassLabelTimepointIndex = bisect.bisect_left(self.timestamps, record['timestamp'])

                            # 2. create a new matrix containing data corresponding to section between old and current class label markers
                            windowLastClass = self.window[:,0:self.currentClassLabelTimepointIndex]
                            classVector = list(itertools.repeat(self.lastClassLabel, self.currentClassLabelTimepointIndex))

                            # 3. add the class label vector to the window
                            windowToDeliver = np.vstack((windowLastClass, classVector))

                            # 4. clear out the accumulated data and timestamp reference list
                            # keep timestamps later than new class label
                            self.timestamps = self.timestamps[self.currentClassLabelTimepointIndex:]

                            if self.debug:
                                print "time to chop window for old class label " + str(self.lastClassLabel) + " before time " + str(record['timestamp'])
                                print "new label is " + str(self.currentClassLabel)
                                print "window to chop is:"
                                print self.window.shape
                                print "timepoint index will be: " + str(self.currentClassLabelTimepointIndex)
                                print "window to deliver will be:"
                                print windowToDeliver.shape
                                print "next window will be:"
                                print self.window[:,self.currentClassLabelTimepointIndex:].shape

                            # 5. keep data associated with new class label
                            self.window = self.window[:,self.currentClassLabelTimepointIndex:]

                            # deliver the window to "data" output
                            windowJson = MatrixToBuffer(windowToDeliver)
                            self.write('data', windowJson)

            elif messageType == self.MESSAGE_TYPE_MATRIX:
                # use this if the input_feature is of type matrix (i.e. window)
                buffer_content = BufferToMatrix(body)
                if self.debug:
                    print "[" + method.consumer_tag + "]"
                    print buffer_content.shape
                    print buffer_content
                    print
                    print





