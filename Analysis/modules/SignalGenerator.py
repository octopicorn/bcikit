__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
import time
import random
import numpy as np
from lib.constants import *
import pandas

"""
This module generates a signal and publishes to the message queue.

This can include the following types of data:
- random numbers (eeg)
- sine waves
- class labels

"""
class SignalGenerator(ModuleAbstract):

    MODULE_NAME = "Signal Generator"

    # __init__ is handled by parent ModuleAbstract

    def setup(self):
        ModuleAbstract.setup(self)

        # time counter, counts number of ticks in current period
        self.period_counter = 0;
        self.lines_counter = 0;
        self.total_counter = 0;

        # params
        # sampling_rate (Hz)
        self.sampling_rate = float(self.module_settings["sampling_rate"]) if "sampling_rate" in self.module_settings else 100.

        # frequency (Hz)
        self.frequency = float(self.module_settings["sine_frequency"]) if "sine_frequency" in self.module_settings else 10.

        # range
        self.range = self.module_settings["range"] if "range" in self.module_settings else [0,1]

        # separator
        self.separator = self.module_settings["separator"] if "separator" in self.module_settings else '\t'

        # skip_lines_prefix
        self.skip_lines_prefix = self.module_settings["skip_lines_prefix"] if "skip_lines_prefix" in self.module_settings else None

        # skip_lines_num
        self.skip_lines = self.module_settings["skip_lines"] if "skip_lines" in self.module_settings else None

        # skip_columns
        self.skip_columns = self.module_settings["skip_columns"] if "skip_columns" in self.module_settings else None

        # data contains timestamp flag
        self.data_already_contains_timestamp = self.module_settings["data_already_contains_timestamp"] if "data_already_contains_timestamp" in self.module_settings else False

        # timestamp_column
        self.timestamp_column = self.module_settings["timestamp_column"] if "timestamp_column" in self.module_settings else 0

        # class_label_column
        self.class_label_column = self.module_settings["class_label_column"] if "class_label_column" in self.module_settings else None

        # flag: whether to playback at the samplig_rate. if false, generate as fast as possible
        self.generate_at_sampling_rate = self.module_settings["generate_at_sampling_rate"] if "generate_at_sampling_rate" in self.module_settings else True

        # pattern
        self.pattern = "rand"
        if self.module_settings["pattern"]:
            self.pattern = self.module_settings["pattern"]

        # what pattern to generate
        if self.pattern == "sine":
            # SINE WAVE PATTERN
            # formula for sine wave is:
            # sine_wave = amp.*sin(2*pi*freq.*time);

            # amplitude = half the distance between the min and max
            amp = abs(float(self.range[1]) - float(self.range[0])) / 2
            # zero will be halfway between the min and max
            offset = float(self.range[1]) - amp

            # np.linspace(-np.pi, np.pi, sampling_rate) --> make a range of x values, as many as sampling rate
            # this is equivalent to 2*pi*time
            sine_x = np.linspace(-np.pi, np.pi, self.sampling_rate)

            self.sine_waves = []
            # sine wave 1
            sine1 = [(amp * np.sin(t * self.sampling_rate/self.frequency)) + offset for t in sine_x]
            self.sine_waves.append(np.tile(sine1,self.frequency))

            # sine wave 2 (double amp, triple freq)
            sine2 = [((2*amp) * np.sin(3 * t * self.sampling_rate/self.frequency)) + offset for t in sine_x]
            self.sine_waves.append(np.tile(sine2,self.frequency))

            # default to the first sine wave (only used if sine)
            self.sine_wave_to_use = 0
            self.generate_pattern_func = "generateSine"

        elif self.pattern == "files":
            # get file list from settings
            self.file_list = self.module_settings["files"]
            # force file list to be list if not already
            if type(self.file_list) != list:
                self.file_list = [self.file_list]

            self.current_file_index = -1
            self.current_file = None

            self.generate_pattern_func = "generateFromFiles"

        else:
            # RANDOM PATTERN
            self.generate_pattern_func = "generateRandom"


        if self.class_label_column:
            self.lastClassLabel = None

        #if self.debug:
        #   print "SAMPLING_RATE: " + str(self.sampling_rate) + " Hz"
        #   print "RANGE: " + str(self.range)

    def getNextFile(self):
        if self.debug:
            print "************* GET NEXT FILE *******************"
        # open the next available file
        self.current_file_index += 1
        # if we have advanced to the next index, and it's bigger than len of file array
        if self.current_file_index >= len(self.file_list):
            # start with the first file again (infinite loop)
            self.current_file_index = 0

        # open the current file
        # print "opening file " + str(self.current_file_index)
        fname = self.file_list[self.current_file_index]
        self.current_file = open(fname)


    def generateSine(self,x):
        message = {"channel_%s" % i: round(self.sine_waves[self.sine_wave_to_use][x],3) for i in xrange(self.num_channels)}
        return message

    def generateFromFiles(self, x):
        message = None
        timestamp = None
        classLabel = None

        # if no file open, open the next one
        if self.current_file is None:
            self.getNextFile()

        # get the next line in file
        nextline = self.current_file.readline()
        if len(nextline) == 0:
            print "------------------------------------------------------------------------------------------------------"
            print "------------------------------------------------------------------------------------------------------"
            print "------------------------------------------------------------------------------------------------------"
            print "------------------------------------------------------------------------------------------------------"
            print "------------------------------------------------------------------------------------------------------"
            print "REACHED END OF FILE"

            self.getNextFile()
            nextline = self.current_file.readline()
            self.lines_counter = 0

        # increment line number
        self.lines_counter += 1

        # Skip line conditions: skip by line number
        # if we are skipping current, just return none
        if self.skip_lines and self.lines_counter <= self.skip_lines:
            return [message, classLabel, timestamp]

        # Skip line conditions: skip by line prefix
        if self.skip_lines_prefix and nextline.startswith(self.skip_lines_prefix):
            print "prefix"
            return [message, classLabel, timestamp]

        # split new line into data by separator
        nextline = np.array(nextline.strip().split(self.separator), dtype=int)

        # TODO deal with edge case of timestamp column coming after class column, may have to use some other strategy besides pop()
        # for now, it's ok to just pop the classLabel first, then timestamp

        # pop the class label off the line if it's present
        if self.class_label_column is not None:
            classLabel = nextline.pop(self.class_label_column)

        # add the timestamp to data if it's not already present
        if self.data_already_contains_timestamp is True:
            # in this case we already have the timestamp, in the 0th position
            timestamp = nextline.pop(self.timestamp_column)
            message = {"channel_%s" % i: float(nextline[i]) for i in xrange(len(nextline))}
            message['timestamp'] = timestamp
            return [message, classLabel, timestamp]

        if self.skip_columns and self.skip_columns > 0:
           skipped_columns = nextline.pop(self.skip_columns-1)

        # just loop through all the elements in the line, assuming each element = 1 channel sample
        message = {"channel_%s" % i: int(float(nextline[i])) for i in xrange(len(nextline))}
        return [message, classLabel, timestamp]

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
            sleep_padding = .30
        elif self.sampling_rate >= 100:
            sleep_padding = .5
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
        self.period_counter = 0
        busy_wait_ticks = 0
        time_start_period = time_start_loop = time.time()

        while(True):

            if self.generate_at_sampling_rate:
                # sleep first
                time.sleep(sleep_length)
                # this gets us most of the way there

            #
            #
            #
            #  now do the processing work
            # generate message by whatever pattern has been specified
            #classLabel = None
            classLabel = None;
            if self.pattern == "files":
                message,classLabel,timestamp = getattr(self,self.generate_pattern_func)(self.period_counter)
            else:
                message = getattr(self,self.generate_pattern_func)(self.period_counter)

            # increment period counter
            self.period_counter += 1
            self.total_counter += 1

            # deliver 'data' output
            if message:
                # generate a timestamp if not already present
                if self.data_already_contains_timestamp is False:
                    timestamp = int(time.time() * 1000000)
                    message['timestamp'] = timestamp
                # PUBLISH message
                self.write('data', message)
                if self.debug:
                    print message

            # deliver 'labels' output
            if classLabel is not None:
                # if class label has changed, generate a new class label message
                # and then update lastClassLabel
                if classLabel != self.lastClassLabel:
                    class_label_message = {"timestamp":timestamp,"class":classLabel}
                    self.write('labels', class_label_message)
                    self.lastClassLabel = classLabel
                    if self.debug:
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print "******************************************************************************"
                        print class_label_message

                if self.debug:
                    print "CLASS: " + classLabel + " TIME: " + timestamp

            if self.generate_at_sampling_rate:
                # now busy wait until we have reached the end of the wait period
                time_elapsed_loop = time.time() - time_start_loop
                while time_elapsed_loop < max_time_per_loop :
                    # busy wait
                    # print time_elapsed_loop
                    time_elapsed_loop = time.time() - time_start_loop
                    busy_wait_ticks = busy_wait_ticks + 1

                # when busy-wait while loop is done, we've reached the end of one loop

            # see how long it took to get our samples per second
            if(self.period_counter == self.sampling_rate):
                time_elapsed_total = time.time() - time_start_period
                # debug message
                if calibration:
                    print str(time_elapsed_total) + " sec to get " + str(self.sampling_rate) + " samples (" + str(busy_wait_ticks) + " busy wait ticks)"

                # reset period counter at end of each period
                self.period_counter = 0

                # alternate sine wave pattern if sine
                if self.pattern == "sine":
                    self.sine_wave_to_use = 1 if self.sine_wave_to_use == 0 else 0

                # reset period timer
                time_start_period = time.time()

            # at end of every loop, reset per-loop timer and ticks counter
            time_start_loop = time.time()
            busy_wait_ticks = 0





