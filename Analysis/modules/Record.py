__author__ = 'odrulea'
from Analysis.modules.ModuleAbstract import ModuleAbstract
from lib.utils import BufferToMatrix
from lib.constants import *
import json
import os.path
from time import localtime,strftime
import numpy as np

"""
This module saves incoming data to durable storage.
"""
class Record(ModuleAbstract):

	MODULE_NAME = "Record"

	# __init__ is handled by parent ModuleAbstract

	def setup(self):
		ModuleAbstract.setup(self)

		self.filename = None
		if 'file' in self.module_conf['outputs']['data']:

			self.save_integers = True

			# get the filename to write to
			self.filename = self.module_conf['outputs']['data']['file']
			file_extension = os.path.splitext(self.filename)[1]

			# include timestamp in the filename if requested
			if self.module_settings['include_timestamp_in_filename']:
				timestamp = strftime("%m-%d-%Y_%H-%M-%S", localtime())
				self.filename = self.filename.replace(file_extension,"-"+timestamp+file_extension)

			# touch the file if overwriting is set to true, this will also set the timestamp
			overwrite_existing = self.module_conf['outputs']['data']['overwrite_existing'] if 'overwrite_existing' in self.module_conf['outputs']['data'] else False

			if overwrite_existing:
				with open(self.filename, 'a'):
					os.utime(self.filename,None)

		# if we don't want the timestamp, pop it off the beginning of the headers
		if not self.module_settings['include_timestamp']:
			self.headers.pop(0)

		if self.debug:
			print self.LOGNAME + "setup"

	def consume(self, ch, method, properties, body):
		"""
		begin looping through the buffer coming in from the message queue subscriber
		"""
		buffer_content = json.loads(body)
		if self.debug:
			print "****************************************************"
			print self.inputs['data']['message_type']

		if self.inputs['data']['message_type'] == MESSAGE_TYPE_TIME_SAMPLE:

			# if the input tag is registered as one of our known inputs from conf.yml
			# use this if the input_feature is an array of json records (i.e. eeg)
			for record in buffer_content:
				# since the channel data is unsorted, we need the headers as a reference to sort the channels
				# timestamp will always come first
				arr = np.array([int(record.get(column_name, None)) for column_name in self.headers])

				line = '\t'.join(str(j) for j in arr) + '\n'
				# write the line to file
				#self.filename_to_write.write(line)

				# UNCOMMENT TO WATCH DATA
				# if self.debug:
				# 	print self.LOGNAME + "incoming data is EEG_RAW"
				# 	print line


		## f.write('hi there\n') # python will convert \n to os.linesep

		elif self.inputs['data']['message_type'] == MESSAGE_TYPE_MATRIX:

			# use this if the input_feature is of type matrix (i.e. window)
			for record in buffer_content:
				# convert the buffer to a matrix object so we can read it
				window = BufferToMatrix(record)

				if self.debug:
					print window.shape

				# a matrix is setup to show channels as rows
				# however, when we save to file, we want to record one time sample per channel on each line
				# since currently time samples are columns in the matrix, we will need to flip the matrix
				# with a transpose, hence the .T notation
				with open(self.filename, "a+", buffering=20*(1024**2)) as myfile:
					for row in window.T:
						# now each row is a time sample instead of a channel
						if self.save_integers:
							line = '\t'.join(str(j) for j in row)
						else:
							line = '\t'.join(str(j) for j in row)
						# write the line to file
						myfile.write(line + '\n')
						# UNCOMMENT TO WATCH DATA
						#if self.debug:
						#	#print window.shape
						#	print line
