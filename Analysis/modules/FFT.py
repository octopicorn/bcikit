__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix, MatrixToBuffer, MatrixToCoords, DTYPE_COORD
import json
import itertools
import numpy as np
from scipy.fftpack import fft
from lib.constants import *

"""
This module runs Fast Fourier Transform (FFT) on an incoming matrix of data, with one FFT per channel (row) in the matrix.
"""
class FFT(ModuleAbstract):

    MODULE_NAME = "FFT"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        if self.debug:
            print self.LOGNAME + "setup"

        # sampling_rate (Hz)
        self.sampling_rate = float(self.module_settings["sampling_rate"]) if "sampling_rate" in self.module_settings else 100.
        self.samples_per_window = self.module_settings["samples_per_window"] if "samples_per_window" in self.module_settings else 500

        if len(self.inputs) and "data" in self.inputs and len(self.outputs) and "data" in self.outputs:
            # check if conversion is supported
            input_message_type = self.inputs["data"]["message_type"]
            input_data_type = self.inputs["data"]["data_type"]
            output_message_type = self.outputs["data"]["message_type"]
            output_data_type = self.outputs["data"]["data_type"]



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
                input_matrix = BufferToMatrix(record)
                [channels,samples] = input_matrix.shape



                # Number of samples
                N = float(samples)
                # Time length represented by each sample (i.e. 1 sec / sampling frequency)
                T = 1.0 / float(self.sampling_rate)

                # The Nyquist limit imposes max 2N+1 frequencies that can be detected by FFT
                fft_plot_length = N/2
                # get a range of x values by applying the Nyquist limit (2N+1) to the number of possible
                # frequencies, based on the sample size and sample frequency
                plot_x = np.linspace(0.0, 1.0/(2.0*T), fft_plot_length)
                window_to_deliver = np.zeros(shape=(channels,fft_plot_length), dtype=DTYPE_COORD)

                #x = np.linspace(0.0, N*T, N)
                ## fake data
                # y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

                for i in np.arange(channels):
                    """
                    loop through channels, and calculate a new FFT plot for each one
                    """
                    # extract sample values for this channel
                    y = input_matrix[i]
                    # calculate fft
                    yf = fft(y)
                    plot_y = np.array(2.0/N * np.abs(yf[0:N/2]))

                    # plot it

                    for j in np.arange(fft_plot_length):
                        window_to_deliver[i,j]["x"] = plot_x[j]
                        window_to_deliver[i,j]["y"] = plot_y[j]

                # encode to buffer for message queue transport
                window_to_deliver = MatrixToBuffer(window_to_deliver, DTYPE_COORD)
                self.write("data",window_to_deliver)

                if self.debug:
                    """
                    debug
                    """
                    # print plot_x[0:5]
                    # print window_to_deliver[0][0:10]
