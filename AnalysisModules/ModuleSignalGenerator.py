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

        # set calibration to True only when you are trying to calibrate the best possible sampling rate
        # accuracy for your system (see comments below)
        calibration = False

        sleep_padding = 1.

        # how many fractions of 1 whole second? that is how long we will sleep
        max_time_per_loop = (1. / float(self.sampling_rate))

        """
        sleep time needs to be reduced to account for processing time
        the factor by which you multiply the sleep time is called "sleep_padding"

        for example:
        a sleep_padding of .61 means that if your max_time_per_loop = 0.002, you
        will sleep for 60% of that time, or 0.00126

        the remainder of the time, the "fine tuned" wait time, will be spent busy-waiting, since that
        is more accurate than sleeping

        we have to account for this sleep_padding factor because we have some processing to do in each loop
        and this takes time, so if we sleep for a full duration of max_time_per_loop, there's no time
        left for the processing

        this is a fine calibration that needs to be run on each individual computer
        with debug set to true, run only this module by itself
        ideally, with the best sleep_padding, you should see as close to 1 sec per sampling_rate # of samples
        like this:
        1.00076007843 sec to get 500.0 samples (1173 busy wait ticks)
        1.00126099586 sec to get 500.0 samples (770 busy wait ticks)
        1.00085878372 sec to get 500.0 samples (1769 busy wait ticks)

        this is a fairly good accuracy, although for lower Hz it can be even more accurate to within 1.000
        """
        # different padding works better for different Hz
        if self.sampling_rate >= 500.:
            sleep_padding = .07
        elif self.sampling_rate >= 375:
            sleep_padding = .30
        elif self.sampling_rate >= 250:
            sleep_padding = .35
        elif self.sampling_rate >= 100:
            sleep_padding = .7
        elif self.sampling_rate >= 50:
            sleep_padding = .79

        if calibration:
            print "********* sampling rate: " + str(self.sampling_rate)

        # sleep will take most but not all of the time per loop
        sleep_length = float(max_time_per_loop * sleep_padding)

        if calibration:
            print "********* max time per loop: " + str(max_time_per_loop)
            print "********* sleep length: " + str(sleep_length)

        # start timer
        self.counter = 0
        busy_wait_ticks = 0
        time_start_period = time_start_loop = time.time()

        while(True):

            # sleep first
            time.sleep(sleep_length)
            # this gets us most of the way there

            # now do the processing work
            # generate message by whatever pattern has been specified
            message = getattr(self,self.generate_pattern_func)(self.counter)
            message['timestamp'] = int(time.time() * 1000000)
            # deliver 'data' output
            self.write('data', message)
            if self.debug:
                print message

            # increment counter
            self.counter += 1

            # now busy wait until we have reached the end of the wait period
            time_elapsed_loop = time.time() - time_start_loop
            while time_elapsed_loop < max_time_per_loop :
                # busy wait
                #print time_elapsed_loop
                time_elapsed_loop = time.time() - time_start_loop
                busy_wait_ticks = busy_wait_ticks + 1

            # when busy-wait while loop is done, we've reached the end of one loop

            # see how long it took to get our samples per second
            if(self.counter == self.sampling_rate):
                time_elapsed_total = time.time() - time_start_period
                # debug message
                if calibration:
                    print str(time_elapsed_total) + " sec to get " + str(self.sampling_rate) + " samples (" + str(busy_wait_ticks) + " busy wait ticks)"

                # reset counter at end of each period
                self.counter = 0

                # alternate sine wave pattern if sine
                if self.pattern == "sine":
                    self.sine_wave_to_use = 1 if self.sine_wave_to_use == 0 else 0

                # reset period timer
                time_start_period = time.time()

            # at end of every loop, reset per-loop timer and ticks counter
            time_start_loop = time.time()
            busy_wait_ticks = 0





