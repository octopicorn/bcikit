__author__ = 'odrulea'
from AnalysisModules.ModuleAbstract import ModuleAbstract
import time
import random
import numpy as np
from lib.constants import *

"""
This module generates a signal and publishes to the message queue.

This can include the following types of data:
- random numbers (eeg)
- sine waves
- class labels

"""
class ModuleSignalGenerator(ModuleAbstract):

    MODULE_NAME = "Signal Generator Module"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        # time counter, counts number of ticks in current period
        self.counter = 0;

        # params
        # sampling_rate (Hz)
        self.sampling_rate = float(self.module_settings["sampling_rate"]) if "sampling_rate" in self.module_settings else 100.

        # frequency (Hz)
        self.frequency = 10.
        if self.module_settings["frequency"]:
            self.frequency = float(self.module_settings["frequency"])

        # range
        self.range = [0,1]
        if self.module_settings["range"]:
            self.range = self.module_settings["range"]

        # pattern
        self.pattern = "rand"
        if self.module_settings["pattern"]:
            self.pattern = self.module_settings["pattern"]

        # what pattern to generate
        if self.pattern == "sine":
            # SINE WAVE PATTERN
            # sine_wave = amp.*sin(2*pi*freq.*time);
            amp = float(self.range[1])
            # np.linspace(-np.pi, np.pi, sampling_rate) --> make a range of x values, as many as sampling rate
            # this is equivalent to 2*pi*time
            sine_x = np.linspace(-np.pi, np.pi, self.sampling_rate/self.frequency)

            self.sine_waves = []
            # sine wave 1
            sine1 = [amp * np.sin(t * self.sampling_rate/self.frequency) for t in sine_x]
            self.sine_waves.append(np.tile(sine1,self.frequency))

            # sine wave 2 (double amp, triple freq)
            sine2 = [(2*amp) * np.sin(3 * t * self.sampling_rate/self.frequency) for t in sine_x]
            self.sine_waves.append(np.tile(sine2,self.frequency))

            # default to the first sine wave (only used if sine)
            self.sine_wave_to_use = 0

            self.generate_pattern_func = "generateSine"
        else:
            # RANDOM PATTERN
            self.generate_pattern_func = "generateRandom"

        #if self.debug:
        #   print "SAMPLING_RATE: " + str(self.sampling_rate) + " Hz"
        #   print "RANGE: " + str(self.range)

    def generateSine(self,x):
        message = {"channel_%s" % i: round(self.sine_waves[self.sine_wave_to_use][x],3) for i in xrange(self.num_channels)}
        return message

    def generateRandom(self,x):
        if self.outputs['data']['data_type'] == DATA_TYPE_RAW_DATA:
            message = {"channel_%s" % i: random.randint(self.range[0],self.range[1]) * random.random() for i in xrange(self.num_channels)}
        elif self.outputs['data']['data_type'] == DATA_TYPE_CLASS_LABELS:
            message = {"class":random.randint(self.range[0],self.range[1])}
        return message

    def generate(self):
        sleep_length = 1 # default to 1 sec delay
        if self.sampling_rate:
            # how many fractions of 1 whole second? that is how long we will sleep
            sleep_length = (1. / float(self.sampling_rate))

        while(True):
            # get message by whatever pattern has been specified
            message = getattr(self,self.generate_pattern_func)(self.counter)

            message['timestamp'] = int(time.time() * 1000000)
            # sleep long enough to get the sampling_rate right
            time.sleep(sleep_length)

            # deliver 'data' output
            self.write('data', message)

            if self.debug:
                print message

            # reset counter at end of each period
            # example, if your sampling_rate is 250Hz, counter goes from 0-249 then resets to 0
            self.counter += 1
            if self.counter >= self.sampling_rate:
                # reached end of one period (1 sec)
                self.counter = 0
                # alternate sine waves if sine pattern
                if self.pattern == "sine":
                    self.sine_wave_to_use = 1 if self.sine_wave_to_use == 0 else 0
