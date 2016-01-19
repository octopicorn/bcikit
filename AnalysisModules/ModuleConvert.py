__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix, MatrixToBuffer, MatrixToCoords, DTYPE_COORD
from lib.lttb import largest_triangle_three_buckets
import json
import itertools
import numpy as np
from lib.constants import *

"""
This module converts from one data type to another.  This was primarily created to for one specific use case, which is
to convert from data matrix to coordinate matrix.  A data matrix has vectors containing pure data, which is optimal for
calculations and transformations.  A coordinate matrix has vectors of [x,y] coordinates, which is optimal for 2D graph
visualizations.

This module is not meant to handle conversion of every type to every type, so an initial mapping of possible conversions
is checked at startup() to make sure a legal conversion is specified in the conf.yml file
"""
class ModuleConvert(ModuleAbstract):

    MODULE_NAME = "Convert Module"

    # LOGNAME is a prefix used to prepaend to debugging output statements, helps to disambiguate messages since the
    # modules run on separate threads
    LOGNAME = "[Analysis Service: Convert Module] "

    # acceptable conversions
    # this dict is structured like this:
    # { "source_type" : ["target_type1","target_type2"]
    SUPPORTED_CONVERSIONS = {
        "RAW_DATA":["RAW_COORDS"]
    }
    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        if self.debug:
            print self.LOGNAME + "setup"

        if len(self.inputs) and "data" in self.inputs and len(self.outputs) and "data" in self.outputs:
            # check if conversion is supported
            input_message_type = self.inputs["data"]["message_type"]
            input_data_type = self.inputs["data"]["data_type"]
            output_message_type = self.outputs["data"]["message_type"]
            output_data_type = self.outputs["data"]["data_type"]

            # check that requested conversion is supported
            if input_message_type == MESSAGE_TYPE_MATRIX:
                if input_data_type not in self.SUPPORTED_CONVERSIONS.keys():
                    raise ValueError(self.LOGNAME + "Conversion from type " + input_data_type + " is not supported.")
                elif output_data_type not in self.SUPPORTED_CONVERSIONS[input_data_type]:
                    raise ValueError(self.LOGNAME + "Conversion from type " + input_data_type + " to " + output_data_type + " is not supported.")
            else:
                raise ValueError(self.LOGNAME + "Input and output 'data' must be of type " + MESSAGE_TYPE_MATRIX + ". Only matrix conversions are supported so far.")

    def consume(self, ch, method, properties, body):
        """
        begin looping through the buffer coming in from the message queue subscriber
        """
        buffer_content = json.loads(body)

        if self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:
            """
            Only MATRIX message type supported for conversions
            """
            # use this if the input_feature is of type matrix (i.e. window)
            for record in buffer_content:
                window_data = BufferToMatrix(record)
                #print self.LOGNAME
                #print type(window_data)
                #print window_data

                window_coords = MatrixToCoords(window_data)
                window_to_deliver = MatrixToBuffer(window_coords, DTYPE_COORD)

                self.write("data",window_to_deliver)

                if self.debug:
                    print self.LOGNAME + "convert from " + str(window_data.shape) + " to " + str(window_coords.shape)
                    print window_coords

