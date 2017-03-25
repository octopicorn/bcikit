__author__ = 'odrulea'
import argparse

from lib.PikaPublisher import PikaPublisher
from lib.PipePublisher import PipePublisher
from lib.devices import get_metrics_names, get_supported_devices, RABBITMQ_ADDRESS, MOCK_DEVICE_ID

_SUPPORTED_DEVICES = get_supported_devices()

from Analysis.modules.ModuleAbstract import ModuleAbstract

"""
Connect to some device
"""


class Connect(ModuleAbstract):
	MODULE_NAME = "Connect Module"

	# LOGNAME is a prefix used to prepaend to debugging output statements, helps to disambiguate messages since the
	# modules run on separate threads
	LOGNAME = "[Analysis Service: Connect Module] "

	# __init__ is handled by parent ModuleAbstract

	def setup(self):
		ModuleAbstract.setup(self)

		"""
		device_name="muse",
        mock_data_enabled=True,
        device_id=MOCK_DEVICE_ID,
        cloudbrain_address=RABBITMQ_ADDRESS,
        buffer_size=10, step_size=10,
        device_port=None,
        pipe_name=None,
        publisher_type="pika",
        device_mac=None
		"""
		if 'mock_data_enabled' not in self.module_settings:
			self.module_settings['mock_data_enabled'] = False

		if self.debug is None:
			self.debug = False

		if 'device_buffer' not in self.module_settings:
			self.module_settings['device_buffer'] = 10

		if 'publisher_type' not in self.module_settings:
			self.module_settings['publisher_type'] = "pika"

		if self.debug:
			print self.LOGNAME + "setup"
			print self.module_settings


	def run(self):

		if self.module_settings['device_name'] == "muse" and not self.module_settings['mock_data_enabled']:
			from cloudbrain.connectors.MuseConnector import MuseConnector as Connector
			if not self.module_settings['device_port']:
				device_port = 9090
		elif self.module_settings['device_name'] in ["openbci","openbci16"] and not self.module_settings['mock_data_enabled']:
			from lib.connectors.OpenBCIConnector import OpenBCIConnector as Connector
		else:
			raise ValueError("Device type '%s' not supported. "
							 "Supported devices are:%s" % (self.module_settings['device_name'], _SUPPORTED_DEVICES))


		metrics = get_metrics_names(self.device_name)

		if self.module_settings['publisher_type'] == "pika":
			publishers = {metric: PikaPublisher(self.module_settings['device_name'],
												self.module_settings['device_id'],
												self.module_settings['cloudbrain_address'],
												metric) for metric in metrics}
		elif self.module_settings['publisher_type'] == "pipe":
			publishers = {metric: PipePublisher(self.module_settings['device_name'],
												self.module_settings['device_id'],
												metric,
												self.module_settings['pipe_name']) for metric in metrics}
		else:
			raise ValueError("'%s' is not a valid publisher type. "
							 "Valid types are %s." % (self.module_settings['publisher_type'], "pika, pipe"))

		for publisher in publishers.values():
			publisher.connect()

		connector = Connector(publishers, self.module_settings['buffer_size'], self.module_settings['step_size'], self.module_settings['device_name'], self.module_settings['device_port'], self.module_settings['device_mac'])
		connector.connect_device()

		if self.module_settings['mock_data_enabled'] and (self.publisher_type != 'pipe'):
			print "INFO: Mock data enabled."

		if self.publisher_type == 'pika':
			print ("SUCCESS: device '%s' connected with ID '%s'\n"
				   "Sending data to cloudbrain at address '%s' ...") % (self.module_settings['device_name'],
																		self.module_settings['device_id'],
																		self.module_settings['cloudbrain_address'])

		try:
			connector.start()
		except KeyboardInterrupt:
			print "CTRL-C: interrupt received, stopping execution"
		finally:
			# clean up
			connector.disconnect()
			print "Connection closed"
			print "Bye"

# def consume(self, ch, method, properties, body):
# 	"""
# 	No-OP
# 	"""
# 	foo = 1

#
#
#
# def validate_opts(opts):
#     """
#     validate that we've got the right options
#
#     @param opts: (list) options to validate
#     @retun opts_valid: (bool) 1 if opts are valid. 0 otherwise.
#     """
#     opts_valid = True
#     if (opts.device_name in ["openbci","openbci_daisy"]) and (opts.device_port is None):
#         opts_valid = False
#     return opts_valid
#
# def get_args_parser():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-i', '--device_id', required=True,
#                         help="A unique ID to identify the device you are sending data from. "
#                              "For example: 'octopicorn2016'")
#
#     parser.add_argument('-m', '--mock', action='store_true', required=False,
#                         help="Use this flag to generate mock data for a "
#                              "supported device name %s" % _SUPPORTED_DEVICES)
#
#
#     parser.add_argument('-n', '--device_name', required=True,
#                         help="The name of the device your are sending data from. "
#                              "Supported devices are: %s" % _SUPPORTED_DEVICES)
#
#
#     parser.add_argument('-c', '--cloudbrain', default=RABBITMQ_ADDRESS,
#                         help="The address of the RabbitMQ instance you are sending data to.\n"
#                              "Use %s to send data to our hosted service. \n Otherwise use "
#                              "'localhost' if running CloudBrain locally" % RABBITMQ_ADDRESS)
#
#
#     parser.add_argument('-o', '--output', default=None,
#                         help="The named pipe you are sending data to (e.g. /tmp/eeg_pipe)\n"
#                              "The publisher will create the pipe.\n"
#                              "By default this is the standard output.")
#
#
#     parser.add_argument('-b', '--buffer_size', default=10,
#                         help='Size of the buffer ')
#
#     parser.add_argument('-s', '--step_size', default=None,
#                         help='Number of samples the chunk advances by (default is equal to buffer size) ')
#
#     parser.add_argument('-p', '--device_port', help="Port used to get data from wearable device.\n"
#                                                                   "Common port values:\n"
#                                                                   "* 9090 for the Muse\n"
#                                                                   "* /dev/tty.usbserial-XXXXXXXX for the OpenBCI")
#
#
#     parser.add_argument('-M', '--device_mac', help="MAC address of device used for Muse connector.")
#
#     parser.add_argument('-P', '--publisher', default="pika",
#                         help="The subscriber to use to get the data.\n"
#                              "Possible options are pika, pipe.\n"
#                              "The default is pika.")
#
#     return parser
#
#
