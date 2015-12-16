__author__ = 'odrulea'

from cloudbrain.utils.metadata_info import get_supported_metrics, get_supported_devices
from cloudbrain.settings import RABBITMQ_ADDRESS, MOCK_DEVICE_ID
import argparse
import imp
import os
import yaml
import time
import threading
import sys

_SUPPORTED_DEVICES = get_supported_devices()
_SUPPORTED_METRICS = get_supported_metrics()

class AnalysisService(object):
    """
    Subscribes and writes data to a file
    Only supports Pika communication method for now, not pipes
    """

    LOGNAME = "[Analysis Service] "

    def __init__(self, device_name, device_id, rabbitmq_address=None, conf_path=None):

        if rabbitmq_address is None:
            raise ValueError(self.LOGNAME + "Pika subscriber needs to have a rabbitmq address!")

        # set vars
        self.device_name = device_name
        self.device_id = device_id
        self.rabbitmq_address = rabbitmq_address
        self.conf_path = conf_path
        self.debug = False

        # local relative filepath, used to load config file and to dynamically load classes
        self.location = os.path.realpath( os.path.join(os.getcwd(), os.path.dirname(__file__)) )

        # this will hold the config yaml info as an array
        self.conf = None

        # an intenral registry of all module threads running
        self.modules = {}

        # setup more vars
        self.setup()

    def setup(self):

        # get config from yaml file (default = ./conf.yml)
        # settings_file_path = os.path.join(self.location, './conf.yml')
        settings_file_path = os.path.join(self.location, self.conf_path)
        stream = file(settings_file_path, 'r')
        # set local conf property from the yaml config file
        self.conf = yaml.load(stream)


    def start(self):

        print self.LOGNAME + "Collecting data ... Ctl-C to stop."

        # loop through each module and start them
        # passing the settings from conf file to each
        if "modules" in self.conf and len(self.conf["modules"]):
            for module_conf in self.conf["modules"]:
                # start each module
                self.launchModule(module_conf, self.device_name, self.device_id, self.rabbitmq_address)
            if self.debug:
                print "-------------------------------------\n\n"
        else:
            print self.LOGNAME + "No modules defined for analysis"

        # this is here so that child threads can run
        while True:
            time.sleep(1)

    def launchModule(self, module_conf, device_name, device_id, rabbitmq_address):

        # module classname is required
        if 'class' in module_conf:
            moduleClassName = module_conf['class']
        else:
            raise ValueError(self.LOGNAME + "ERROR: class not defined for module: " + str(module_conf))

        # get module parameters for any operation at the service level (optional)
        if 'settings' in module_conf:
            module_settings = module_conf['settings']
            # debug output
            if 'debug' in module_settings and module_settings['debug'] == True:
                # if any of the modules have debug turned on, turn on the service debug too
                self.debug = True
                print "-------------------------------------\n" \
                     "" + self.LOGNAME + "STARTING...\n" \
                     "Module: " + moduleClassName + "\n" \
                    "Configuration: " + str(module_conf) + "\n"

        module_id = None
        if 'id' in module_conf:
            module_id = module_conf['id']

        # dynamically import the module
        module_filepath = os.path.join(self.location, moduleClassName+'.py')
        py_mod = imp.load_source(moduleClassName, module_filepath)

        # instantiate the imported module
        moduleInstance = getattr(py_mod, moduleClassName)(device_name=device_name, device_id=device_id,
                                                     rabbitmq_address=rabbitmq_address, module_conf=module_conf,
                                                     global_conf=self.conf)


        # all modules should implement start() and stop()
        thread = threading.Thread(target=moduleInstance.start)
        thread.daemon = True
        # assign the thread to internal registry and start it up
        self.modules[module_id] = thread
        self.modules[module_id].start()


        #moduleInstance.stop()



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--device_id', required=True,
                        help="A unique ID to identify the device you are sending data from. "
                             "For example: 'octopicorn2015'")
    parser.add_argument('-n', '--device_name', required=True,
                        help="The name of the device your are sending data from. "
                             "Supported devices are: %s" % _SUPPORTED_DEVICES)
    parser.add_argument('-c', '--cloudbrain', default=RABBITMQ_ADDRESS,
                        help="The address of the CloudBrain instance you are sending data to.\n"
                             "Use " + RABBITMQ_ADDRESS + " to send data to our hosted service. \n"
                                                         "Otherwise use 'localhost' if running CloudBrain locally")

    opts = parser.parse_args()

    return opts


def main():
    opts = parse_args()

    device_name = opts.device_name
    device_id = opts.device_id
    cloudbrain_address = opts.cloudbrain

    run(device_name,
        device_id,
        cloudbrain_address
        )

def run(device_name='muse',
        device_id=MOCK_DEVICE_ID,
        cloudbrain_address=RABBITMQ_ADDRESS
        ):

    service = AnalysisService(device_name=device_name,
                          device_id=device_id,
                          rabbitmq_address=cloudbrain_address
                          )
    service.start()


if __name__ == "__main__":
    main()


