__author__ = 'odrulea'

import json
import numpy as np
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import MatrixToBuffer

"""
The Windows Module is meant for bounding raw EEG data into epochs.
The current implementation is a "rolling" window of fixed width.  This is typically used in the testing or "online"
usage of BCI.  This type of window is characterized by a fixed width, and an overlap parameter. The idea is that, as
live data is coming in, you are constantly checking it against a trained model. This could also be used as a sort of
simple buffering on raw data, if you set the overlap to 0 for example.
"""

class TimeWindow(ModuleAbstract):
    
    MODULE_NAME = "Windows"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        super(TimeWindow,self).setup()

        # init self vars
        # window params
        self.samples_per_window = self.module_settings["samples_per_window"] if "samples_per_window" in self.module_settings else 500
        self.window_overlap = self.module_settings["window_overlap"] if "window_overlap" in self.module_settings else 100
        self.multiplier = self.module_settings["multiplier"] if "multiplier" in self.module_settings else None

        if self.debug:
            print self.LOGNAME + "Samples per window:" + str(self.samples_per_window)
            print self.LOGNAME + "Window overlap:" + str(self.window_overlap)

        # create a blank matrix of zeros as a starting window
        self.window = np.matrix(np.zeros((self.num_channels, self.samples_per_window)))
        # create a blank matrix of zeros as the rolling overlap window
        if self.window_overlap > 0:
            self.nextWindowSegment = np.matrix(np.zeros((self.num_channels, self.window_overlap)))

        # define range based on overlap length, this will be used in loop below
        self.trimOldWindowDataIndexRange = np.arange(self.window_overlap)

        self.plotActive = True
        self.windowFull = False
        self.fill_counter = 0
        self.rolling_counter = 0




    def consume(self, ch, method, properties, body):
        """
        Windows Module chops streaming multi-channel time series data into 'windows'
        Semantically, window = epoch = trial = matrix
        As matrix, window has dimensions [rows, cols] - standard notation for numpy, matlab, etc

        Each row vector captures data per EEG channel
        Each column vector captures data per time-point

        self.window is the main window
        # vector representing only channel 2 data for the entire time series in the window:
        self.window[1,:]
        # vector representing all channels' data at timepoint 10:
        self.window[:,9]
        # vector representing only channel 2 data between time 10 and 50:
        self.window[1,9:50]]
        # (note that range endpoint is not included in slice, so that's why it's 50 and not 49)

        Visualization
        You could use this matrix to plot n-channels number of streaming graphs,
        or you could superimpose n-channels number of lines on the same streaming graph
        """

        # begin looping through the buffer coming in from the message queue subscriber
        # because of cloudbrain connector publisher convention, this is assumed to be in json format
        # note: when using pika, after retrieving json, keys are always in utf-8 format
        buffer_content = json.loads(body)
        for record in buffer_content:

            #print buffer_content
            # get the next data out of the buffer as an array indexed by column names
            # arr = array of all column names
            arr = np.array([record.get(column_name, None) for column_name in self.headers])
            if self.windowFull is False:
                # window is not full yet
                # just keep collecting data into main window until we have filled up the first one
                # i.e. write next data column in matrix
                # note: timestamp is not used (i.e. we skip arr[0])
                # note: we have use transpose (T) property to get the horizontal array coming from rabbitmq into a
                # vertical column for use in our matrix
                self.window[:, self.fill_counter] = arr[1:len(self.headers)][np.newaxis].T

                self.fill_counter = self.fill_counter + 1
                #print "still filling up first window: " + str(self.fill_counter) + " rows so far"

                # once we've reached one full window length, set the flag windowFull to true so we can begin rolling
                if self.fill_counter == self.samples_per_window:
                    self.windowFull = True
                    # send the window data (first window is sent)
                    self.sendData()

                    if self.window_overlap == 0:
                        # if there is 0 overlap, we will just reset initial loop and do this over and over
                        self.windowFull = False
                        self.fill_counter = 0

                    else:
                        if self.debug:
                            print self.LOGNAME + "Received first window of " + str(self.samples_per_window) + " samples:\n"


            else:
                # accumulate every new data into next window segment
                # note: we have use transpose (T) property to get the horizontal array coming from rabbitmq into a
                # vertical column for use in our matrix
                self.nextWindowSegment[:, self.rolling_counter] = arr[1:len(self.headers)][np.newaxis].T

                # keep incrementing rolling counter
                self.rolling_counter = self.rolling_counter + 1

                # check if we have reached next window yet
                if(self.rolling_counter >= self.window_overlap):
                    # reached overlap, time to roll over to next window

                    # Step 1: trim off old data columns from the beginning of window
                    self.window = np.delete(self.window, self.trimOldWindowDataIndexRange, 1)
                    # Step 2: append next window segment columns onto the front (right) of window
                    self.window = np.hstack((self.window, self.nextWindowSegment))

                    # we've got a new window to deliver, time to publish it
                    self.sendData()

                    # since we've rolled to a new window, time to reset the rolling counter
                    self.rolling_counter = 0

    def sendData(self):
        if(self.multiplier):
            self.window = self.multiplier * self.window

        windowJson = MatrixToBuffer(self.window)
        self.write('data', windowJson)

        # debug
        if self.debug:
            print self.window.shape
            print self.window
