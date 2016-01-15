__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
from lib.constants import *
import json
import bisect
import itertools
import numpy as np

"""
This module saves incoming data to durable storage.
"""
class ModuleRecord(ModuleAbstract):

    MODULE_NAME = "Record Module"

    # LOGNAME is a prefix used to prepaend to debugging output statements, helps to disambiguate messages since the
    # modules run on separate threads
    LOGNAME = "[Analysis Service: Record Module] "

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        self.file_to_write = None
        if 'file' in self.module_conf:
            # set file write mode ('a', or append, by default)
            file_write_mode = self.module_conf['file_write_mode'] if 'file_write_mode' in self.module_conf else 'a'
            self.file_to_write = open(self.module_conf['file'],file_write_mode)

        # f.close() # you can omit in most cases as the destructor will call it

        if self.debug:
            print self.LOGNAME + "setup"

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == MESSAGE_TYPE_TIME_SAMPLE:
            print self.LOGNAME + "incoming data is EEG_RAW"
            # if the input tag is registered as one of our known inputs from conf.yml
            # use this if the input_feature is an array of json records (i.e. eeg)
            for record in buffer_content:
                arr = np.array([int(record.get(column_name, None)) for column_name in self.headers])
                line = '\t'.join(str(j) for j in arr)
                self.file_to_write
                if self.debug:
                    print line# record
                    print "---------------------------------------"


        ## f.write('hi there\n') # python will convert \n to os.linesep

        elif self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:
            print self.LOGNAME + "incoming data is MATRIX"
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                window = BufferToMatrix(record)

                if self.debug:
                    print "[" + method.consumer_tag + "]"
                    print window.shape
                    #print window
                    print
                    print





