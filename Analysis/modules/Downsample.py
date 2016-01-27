__author__ = 'odrulea'

from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix, MatrixToBuffer, DTYPE_COORD
from lib.lttb import largest_triangle_three_buckets
from lib.constants import *
import numpy as np
import json

class Downsample(ModuleAbstract):

    MODULE_NAME = "Downsample Module"
    LOGNAME = "[Analysis Service: Downsample Module] "

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        super(Downsample, self).setup()

        if self.inputs['data']['message_type'] != MESSAGE_TYPE_MATRIX:
            raise ValueError(self.LOGNAME + "Input 'data' must be of type " + MESSAGE_TYPE_MATRIX)

        self.counter = 0

        self.formula = "lttb" # default: LTTB formula
        if "formula" in self.module_settings:
            self.formula = self.module_settings["formula"]

        self.percent = 10    # default: downsample to 10% or original size
        if "percent" in self.module_settings:
            self.percent = self.module_settings["percent"]

    def consume(self, ch, method, properties, body):
        """
        Downsample Module does exactly what it says
        """

        # begin looping through the buffer coming in from the message queue subscriber
        #print body
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                window_coords = BufferToMatrix(record)
                [channels, dataset_length] = window_coords.shape

                if self.formula == "lttb":
                    # find target threshold (i.e. size of data after downsmapling),
                    # by applying percent setting to dataset size
                    threshold = int(dataset_length * (float(self.percent) / 100.))

                window_coords_downsampled = np.matrix([np.array(largest_triangle_three_buckets(row,threshold)) for row in window_coords])
                window_to_deliver = MatrixToBuffer(window_coords_downsampled, DTYPE_COORD)
                self.write('data', window_to_deliver)

                if self.debug:
                    print self.LOGNAME + "[" + method.consumer_tag + "]"
                    print "original window dimensions: " + str(window_coords.shape)
                    print "downsampled window dimensions: " + str(window_coords_downsampled.shape)
                    #print window_coords_downsampled
                    #print window_to_deliver
                    #print len(window_to_deliver)


        elif self.inputs['data']['message_type'] == MESSAGE_TYPE_TIME_SAMPLE:
            for record in buffer_content:
                """
                record is a dict type object
                """
                # get the nth next data out of the buffer
                # output is an array indexed by column names, i.e. one datapoint per channel
                if(self.counter % self.factor == 0):
                    # pass through
                    self.write('data',record)
                    if self.debug:
                        print record

                    self.counter = 0
                else:
                    if self.debug:
                        print "----skip----"
                self.counter = self.counter + 1




