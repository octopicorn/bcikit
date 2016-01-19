__author__ = 'odrulea'
from abc import ABCMeta, abstractmethod
from lib.PikaSubscriber import PikaSubscriber
#from cloudbrain.subscribers.PikaSubscriber import PikaSubscriber
from cloudbrain.publishers.PikaPublisher import PikaPublisher
from cloudbrain.utils.metadata_info import get_num_channels
import lib.constants

class ModuleAbstract(object):
    __metaclass__ = ABCMeta

    MODULE_NAME = "Abstract"

    def __init__(self, device_name, device_id, rabbitmq_address, module_conf={}, global_conf={}):
        """
        global constructor for all module classes, not meant to be overwritten by subclasses
        :param device_name:
        :param device_id:
        :param rabbitmq_address:
        :param moduleConf:
        :return:
        """

        # LOGNAME is a prefix used to prepaend to debugging output statements, helps to disambiguate messages since the
        # modules run on separate threads
        self.LOGNAME = "[Analysis Service: " + self.MODULE_NAME + "] "

        # set global properties common to all
        self.device_name = device_name
        self.device_id = device_id
        self.rabbitmq_address = rabbitmq_address
        self.module_conf = module_conf
        self.global_conf = global_conf
        self.global_settings = self.global_conf["global"] if self.global_conf and "global" in self.global_conf else {}

        # id
        self.id = None
        if 'id' in self.module_conf:
            self.id = self.module_conf['id']

        # inputs (optional)
        self.inputs = {}
        if 'inputs' in self.module_conf:
            self.inputs = self.module_conf['inputs']

        # outputs (optional)
        self.outputs = {}
        if 'outputs' in self.module_conf:
            self.outputs = self.module_conf['outputs']

        # module parameters (optional)
        self.module_settings = None
        if 'settings' in self.module_conf:
            self.module_settings = self.module_conf['settings']

        self.subscriber = None
        self.publishers = {}
        self.output_buffers = {}


        self.num_channels = 0
        self.headers = []

        # debug
        self.debug = False
        if 'debug' in self.module_settings:
            if self.module_settings['debug'] is True:
                self.debug = True
                # if len(self.inputs):
                #     print "inputs:" + str(self.inputs)
                # if len(self.outputs):
                #     print "outputs:" + str(self.outputs)

        # call setup()
        self.setup()


    def setup(self):
        """
        Generic setup for any analysis module, can be overriden by implementing in any child class
        This sets up subscriber and publisher based on input and output feature names
        """
        # usually this module is used with incoming EEG,
        # so we'd like to know num channels, and a header is for convenience
        # hard-coded "eeg" could be a problem if the device's metric name for raw data is not "eeg"
        # currently "eeg" is a known good metric name for both OpenBCI and Muse
        self.num_channels = get_num_channels(self.device_name,"eeg")
        # overridden by global setting
        if "num_channels" in self.global_settings:
            self.num_channels = self.global_settings["num_channels"]
        # overridden by module specific setting
        if "num_channels" in self.module_settings:
            self.num_channels = self.module_settings["num_channels"]

        self.headers = ['timestamp'] + ['channel_%s' % i for i in xrange(self.num_channels)]

        # if input, instantiate subscriber
        if len(self.inputs):
            # there is only one subscriber to handle all inputs
            self.subscriber = PikaSubscriber(device_name=self.device_name,
                                                     device_id=self.device_id,
                                                     rabbitmq_address=self.rabbitmq_address,
                                                     metrics=self.inputs)

        # if output, instantiate publishers
        if len(self.outputs):

            for output_key, output in self.outputs.iteritems():
                # each output has a specific key, assign a placeholder for it in publishers collection
                self.publishers[output_key] = {}
                self.output_buffers[output_key] = {}

                # each output has a parameter called "message_queues"
                # this can be a single value or a list, i.e. "foo" or ["foo1","foo2"]
                # most of the time, this will be a single string
                # an example of where an output might use more than one message_queue might be:
                # one output goes to visualization, while a second copy continues down the processing chain

                if 'message_queues' in output:
                    # for convenience, convert the "message_queues" parameter to list if it isn't already
                    if type(output['message_queues']) != list:
                        output['message_queues'] =  [output['message_queues']]

                    # there is one publisher per output
                    for message_queue_name in output['message_queues']:
                        self.publishers[output_key][message_queue_name] = PikaPublisher(
                                                                    device_name=self.device_name,
                                                                    device_id=self.device_id,
                                                                    rabbitmq_address=self.rabbitmq_address,
                                                                    metric_name=message_queue_name)

                        # also instantiate an output buffer for each publisher
                        self.output_buffers[output_key][message_queue_name] = []


    def start(self):
        """
        Consume and write data to file
        :return:
        """

        # unleash the hounds!
        if len(self.publishers):
            if self.debug:
                print "[" + self.MODULE_NAME + "] starting publishers"

            for output_key, output_message_queues in self.publishers.iteritems():
                for output_message_queue, publisher in output_message_queues.iteritems():
                    publisher.connect()
                    #if self.debug:
                    #    print self.LOGNAME + " publisher [" + output_key + "][" + output_message_queue + "] started streaming"
                    print self.LOGNAME + " publisher [" + self.device_id + ":" + self.device_name + ":" + output_message_queue + "] started streaming"

            if self.subscriber is None:
                # if there are publishers but no subscribers defined, this is a generator type module
                # so we need some way to start publishing messages without depending on the consume() method
                self.generate()

        # it begins!
        if self.subscriber and self.debug:
            print "[" + self.MODULE_NAME + "] starting subscribers"

        self.subscriber.connect()
        self.subscriber.consume_messages(self.consume)

        return

    def stop(self):
        """
        Unsubscribe and close file
        :return:
        """
        print "Abstract: stopped"
        self.subscriber.disconnect()

    def generate(self):
        """
        generate to the message queue
        :return:
        """
        print "Abstract: generate"

    def consume(self, ch, method, properties, body):
        """
        consume the message queue from rabbitmq
        :return:
        """
        print "Abstract: consume"

    def write(self, output_key, datum):
        """
        deliver one message (buffered)
        """
        for message_queue_name in self.output_buffers[output_key].keys():
            # add one data point to each buffer defined for this output
            self.output_buffers[output_key][message_queue_name].append(datum)

            # if buffer is full, publish it, using appropriate publisher
            if len(self.output_buffers[output_key][message_queue_name]) >= self.outputs[output_key]['buffer_size']:
                # publishing means deliver the buffer (which is now full) to the publisher
                # this sends the entire buffer onto message queue
                self.publishers[output_key][message_queue_name].publish(self.output_buffers[output_key][message_queue_name])
                # and then reset the buffer
                self.output_buffers[output_key][message_queue_name] = []

