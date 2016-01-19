__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
from lib.constants import *
import json
from time import localtime,strftime
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
        if 'file' in self.module_conf['outputs']['data']:
            # set file write mode ('a', or append, by default)
            file_write_mode = self.module_conf['outputs']['data']['file_write_mode'] if 'file_write_mode' in self.module_conf['outputs']['data'] else 'a'

            # get the filename to write to
            filename = self.module_conf['outputs']['data']['file']

            # include timestamp in the filename if requested
            if self.module_settings['include_timestamp_in_filename']:
                timestamp = strftime("_%H-%M-%S", localtime())
                filename = filename.replace('.data',timestamp+'.data')

            # open the file
            self.file_to_write = open(filename,file_write_mode)

        # if we don't want the timestamp, pop it off the beginning of the headers
        if not self.module_settings['include_timestamp']:
            self.headers.pop(0)

        if self.debug:
            print self.LOGNAME + "setup"

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == MESSAGE_TYPE_TIME_SAMPLE:

            # if the input tag is registered as one of our known inputs from conf.yml
            # use this if the input_feature is an array of json records (i.e. eeg)
            for record in buffer_content:
                # since the channel data is unsorted, we need the headers as a reference to sort the channels
                # timestamp will always come first
                arr = np.array([int(record.get(column_name, None)) for column_name in self.headers])

                line = '\t'.join(str(j) for j in arr) + '\n'
                # write the line to file
                self.file_to_write.write(line)

                if self.debug:
                    print self.LOGNAME + "incoming data is EEG_RAW"
                    print line
                    print "---------------------------------------"


        ## f.write('hi there\n') # python will convert \n to os.linesep

        elif self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:

            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                # convert the buffer to a matrix object so we can read it
                window = BufferToMatrix(record)
                if self.debug:
                    print self.LOGNAME + "incoming data is MATRIX " + str(window.shape)
                    #print window

                # a matrix is setup to show channels as rows
                # however, when we save to file, we want to record one time sample per channel on each line
                # since currently time samples are columns in the matrix, we will need to flip the matrix
                # with a transpose, hence the .T notation
                for row in window.T:
                    # now each row is a time sample instead of a channel
                    line = '\t'.join(str(j) for j in row) + '\n'
                    # write the line to file
                    self.file_to_write.write(line)

                    if self.debug:
                        print line
                        print "---------------------------------------"


# f.close() # you can omit in most cases as the destructor will call it


