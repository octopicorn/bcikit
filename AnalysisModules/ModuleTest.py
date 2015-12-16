__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
import json
import bisect
import itertools
import numpy as np

"""
This module does nothing.  Meant to be used as a blank template to start new modules from.
Shows you the basic 2 methods to implement: setup() and consume()
If you are publishing an output, when you are ready to send it to mq, use self.write() at the end of consume()
"""
class ModuleTest(ModuleAbstract):

    MODULE_NAME = "Test Module"

    # LOGNAME is a prefix used to prepaend to debugging output statements, helps to disambiguate messages since the
    # modules run on separate threads
    LOGNAME = "[Analysis Service: Test Module] "

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        if self.debug:
            print self.LOGNAME + "setup"

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == self.MESSAGE_TYPE_TIME_SAMPLE:
            # if the input tag is registered as one of our known inputs from conf.yml
            # use this if the input_feature is an array of json records (i.e. eeg)
            for record in buffer_content:
                if self.debug:
                    print record

        elif self.inputs['data']['message_type'] == self.MESSAGE_TYPE_MATRIX:
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                window = BufferToMatrix(record)

                if self.debug:
                    print "[" + method.consumer_tag + "]"
                    print window.shape
                    print window
                    print
                    print



        #self.countTotal += 1

        # if(self.countTotal > 100):
        #     self.actualRatio = float(float(self.countInput1) / float(self.countInput2))
        #     print "ratio is: " + str(self.actualRatio)
        #     self.countInput1 = 0
        #     self.countInput2 = 0
        #     self.countTotal = 0




