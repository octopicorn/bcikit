import time

from lib.connectors.ConnectorInterface import Connector
from lib.connectors.OpenBCIBoard import OpenBCIBoard
from lib.devices import get_num_channels
from lib.utils import is_odd


class OpenBCIConnector(Connector):
	def __init__(self, publishers, buffer_size, step_size, device_type='openbci',
	             device_port='/dev/tty.usbserial-DN0096FW', device_mac=None):
		"""
		:return:
		"""
		super(OpenBCIConnector, self).__init__(publishers, buffer_size, step_size, device_type, device_port, device_mac)

		# this is used only for the 16 channel Daisy module
		self.lower8ChannelsSample = None


	def connect_device(self):
		"""
		:return:
		"""
		num_channels = get_num_channels(self.device_name, 'eeg')
		self.device = OpenBCIBoard(port=self.device_port, num_channels=num_channels)

	def start(self):
		# callback functions to handle the sample for that metric (each metric has a specific number of channels)
		cb_functions = {metric: self.callback_factory(metric, get_num_channels(self.device_name, metric))
		                for metric in self.metrics}

		self.device.start(cb_functions)

	def stop(self):
		self.device.stop()

	def disconnect(self):
		self.device.disconnect()

	def callback_factory(self, metric_name, num_channels):
		"""
		Callback function generator for OpenBCI metrics
		:return: callback function
		"""

		def callback(sample):
			"""
			Handle OpenBCI samples for that metric
			:param sample: the sample to handle

			NOTE: 16 Channel system (Daisy Module)

			The daisy module increases the channels to 16. However, this is achieved by sending alternating packets from the 8 base channels and then the 8 daisy module channels.

			For further reading: http://docs.openbci.com/software/02-OpenBCI_Streaming_Data_Format
			"Our 16 channel system allows for control of individual settings for all 16 channels, and data is retrieved from both ADS1299 IC at a rate of 250SPS. The current bandwith limitations on our serial radio links limit the number of packets we can send. To solve for this, we are sending data packet at the same rate of 250SPS, and alternating sample packets between the on Board ADS1299 and the on Daisy ADS1299. The method takes an average of the current and most recent channel values before sending to the radio. On odd sample numbers, the Board ADS1299 values are sent, and on even sample numbers, the Daisy ADS1299 samples are sent. When running the system with 16 channels, it is highly recommended that you use an SD card to store the raw (un-averaged) data for post processing."


			"""
			if(num_channels == 8):
				"""
				In the 8 channel version, write data to buffer as it comes in
				"""
				message = {}
				for i in range(8):
					channel_value = "%.4f" % (sample.channel_data[i] * 10 ** 12)  # Nano volts
					message["channel_%s" % i] = channel_value
					message['timestamp'] = int(time.time() * 1000000)  # micro seconds

				self.buffers[metric_name].write(message)
			elif(num_channels == 16):
				"""
				In the 16 channel version, we have to combine every odd and even sample into a unified sample
				to write to buffer
				"""

				if(not is_odd(sample.id)):
					"""
					EVEN sample - comes from base board ADS1299
					"""
					message = {}
					for i in range(8):
						channel_value = "%.4f" % (sample.channel_data[i] * 10 ** 12)  # Nano volts
						message["channel_%s" % i] = float(channel_value)

					# save the first sample half
					self.lower8ChannelsSample = message

				else:
					"""
					ODD sample - comes from Daisy module ADS1299
					merge with half sample stored in self.lower8ChannelsSample
					"""
					if self.lower8ChannelsSample is not None:
						message = {}
						for i in range(8):
							channel_value = "%.4f" % (sample.channel_data[i] * 10 ** 12)  # Nano volts
							message["channel_%s" % int(i+8)] = float(channel_value)
							message['timestamp'] = int(time.time() * 1000000)  # micro seconds

						merged_samples = dict(self.lower8ChannelsSample, **message)
						self.lower8ChannelsSample = None

						#print merged_samples
						self.buffers[metric_name].write(merged_samples)

		return callback
