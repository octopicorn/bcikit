__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix, MatrixToBuffer
import lib.constants as constants
import json
import bisect
import itertools
import numpy as np
from scipy.signal import lfilter as lfilter
from lib.utils import FilterCoefficients

"""
This module applies two filters to the data coming through:
- Notch Filter
- Bandpass Filter
If you are publishing an output, when you are ready to send it to mq, use self.write() at the end of consume()
"""
class Filter(ModuleAbstract):

    MODULE_NAME = "Filter"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        if self.debug:
            print self.LOGNAME + "setup"

        # notch
        self.notch_filter = self.module_settings["notch_filter"] if "notch_filter" in self.module_settings and self.module_settings["notch_filter"] is not False else None

        # bandpass
        self.bandpass_filter = self.module_settings["bandpass_filter"] if "bandpass_filter" in self.module_settings and self.module_settings["bandpass_filter"] is not False else None

        # create the notch filter coefficients (60 Hz)
        """
        For more info, see http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html#scipy.signal.butter
        Or http://www.mathworks.com/help/signal/ref/butter.html?s_tid=gn_loc_drop#buct3_m
        """
        if self.notch_filter:
            self.notch_filter_b, self.notch_filter_a = FilterCoefficients('bandstop',250.0,np.array(self.notch_filter))

        # create the bandpass filter (7-13Hz)
        if self.bandpass_filter:
            self.bandpass_filter_b, self.bandpass_filter_a = FilterCoefficients('bandpass',250.0,np.array(self.bandpass_filter))

    def filterWindow(self, window, filter_func):
        """
        apply filter to each row (channel) in the window
        """
        return np.array([getattr(self,filter_func)(channel_data) for channel_data in window], dtype=np.int)

    def meanFilter(self, channel_data):
        """
        A common step, remove the mean.  This has the effect of centering the data around the mean value.
        """
        return channel_data - channel_data.mean()

    def notchFilter(self, channel_data):
        """
        For more info, see http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.lfilter.html
        """
        return lfilter(self.notch_filter_b, self.notch_filter_a, channel_data)

    def bandpassFilter(self, channel_data):
        """
        """
        return lfilter(self.bandpass_filter_b, self.bandpass_filter_a, channel_data)

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == constants.MESSAGE_TYPE_TIME_SAMPLE:
            # if the input tag is registered as one of our known inputs from conf.yml
            # use this if the input_feature is an array of json records (i.e. eeg)
            for record in buffer_content:
                if self.debug:
                    print record

        elif self.inputs['data']['message_type'] == constants.MESSAGE_TYPE_MATRIX:
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                window = BufferToMatrix(record)
                # print "window ",window

                # apply notch filter
                if self.notch_filter:
                    window = self.filterWindow(window,'notchFilter')
                    #print "notched " + str(filtered_window.shape)

                # apply bandpass filter
                if self.bandpass_filter:
                    window = self.filterWindow(window,'bandpassFilter')
                    #print "bandpass " + str(filtered_window.shape)

                # remove mean
                window = self.filterWindow(window,'meanFilter')
                # print "mean removed ", window
                # publish results
                windowJson = MatrixToBuffer(np.matrix(window))
                self.write('data', windowJson)

                if self.debug:
                    """
                    """
                    print "--------------------------------------------"
                    print window.shape
                    print
