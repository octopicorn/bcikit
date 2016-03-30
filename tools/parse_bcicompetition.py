# combine CNT and MRK files into single csv

import argparse
import csv
import glob
import math
import os
import pandas


"""
This is a utility file that can be used to parse datasets in which raw data is separate from class labels, following the model established by the BCI Competition IV

Example usage:
python tools/parse_bcicompetition.py -d /path/to/train_data

There is a flag -m used to specify "mode"
There are 3 possible values:
train - used to parse train data files (where "_mrk" type files exist)
test - used to parse test data files (where "_mrk" type files exist)
testprint - used to output to terminal output the content needed to convert "true_y" result files to "_mrk" files in preparation for "test" mode above
"""

class MRKParser(object):
	def __init__(self, fname, class_length, mode="train"):
		self.mrk_file_class_length = class_length
		self.mrk_file_mode = mode


		self.mrk_file_class_length_counter = 0
		self.mrk_file_current_class = None

		if self.mrk_file_mode == "train":
			mrk_fname = fname.replace("_cnt", "_mrk")
		elif self.mrk_file_mode in ["test","testprint"]:
			# if you're parsing test files, make sure have moved the "true_y" files into the same dir as the "eval" files
			mrk_fname = fname.replace("_cnt", "_true_y")
		# open that file
		self.mrk_file_data = pandas.read_table(mrk_fname, header=None)

		self.mrk_file_data_len = len(self.mrk_file_data)
		self.mrk_file_next_class = None
		self.mrk_file_next_index = None
		self.mrk_file_next_row = None
		print "MRK file loaded", "with mode:", self.mrk_file_mode, "classLength:", self.mrk_file_class_length

	def getClassLabelFromMrkFile(self, currentLineIndex):
		"""
		line number passed in is based on an index starting with 1 (i.e. first line is 1, not 0)
		"""

		# in case of test mrk files, we just return the class line by line, no logic needed
		if self.mrk_file_mode in ["test","testprint"]:
			classLabel = self.mrk_file_data[0][currentLineIndex]
			# these files have NaN so convert those to 0 to normalize it
			if math.isnan(float(classLabel)):
				classLabel = 0
			classLabel = int(classLabel)

			if self.mrk_file_mode == "testprint" and classLabel != 0 and classLabel != self.mrk_file_current_class:
				print str(currentLineIndex/10) + "\t" + str(classLabel)

			self.mrk_file_current_class = int(classLabel)
			return classLabel



		if self.mrk_file_next_index is None or self.mrk_file_class_length_counter >= self.mrk_file_class_length:
			"""
			if we've either 1) just started new file, or 2) reached end of current class:
			- revert/init class to default of 0, this class label will be broadcast until next class is detected
			- get next class entry
			- reset counter to 0
			"""

			# reset class length counter and
			self.mrk_file_current_class = 0
			self.mrk_file_class_length_counter = 0

			if self.mrk_file_next_index is None:
				self.mrk_file_next_row = 0
			else:
				self.mrk_file_next_row += 1

			# unless we've reached the end of the MRK data,
			# peek ahead to see at what time index will be the next class label
			if self.mrk_file_next_row < self.mrk_file_data_len:
				self.mrk_file_next_index = self.mrk_file_data[0][self.mrk_file_next_row]
				self.mrk_file_next_class = self.mrk_file_data[1][self.mrk_file_next_row]

			print "parsing class tag", self.mrk_file_next_row, self.mrk_file_next_class

		if currentLineIndex >= self.mrk_file_next_index:
			"""
			if the line number passed in hit the next class starting point, we've just started that next class
			"""
			self.mrk_file_current_class = int(self.mrk_file_next_class)


		# only increment counter if we're currently "in" an actual class (i.e. 	not 0)
		if self.mrk_file_current_class is not 0:
			self.mrk_file_class_length_counter += 1

		return self.mrk_file_current_class


def parseFile(input_fname, output_filepath, mode, classLength):
	if mode == "testprint":
		active = False
	else:
		active = True

	# load
	mrkParser = MRKParser(input_fname, classLength, mode)

	if active:
		# overwrite target file if already exists
		if os.path.isfile(output_filepath):
			print "Output file cleared"
			os.remove(output_filepath)

		# open target file for writing
		output_file = open(output_filepath, 'w')
		csv_writer = csv.writer(output_file, delimiter='\t')
		print "Output file created", output_filepath

	# open the input file
	with open(input_fname, 'rb') as input_file:
		cnt_reader = csv.reader(input_file, delimiter='\t')

		# loop through the input file
		i = 0
		for row in cnt_reader:
			row.append(str(mrkParser.getClassLabelFromMrkFile(i)))
			# write to output file
			if active:
				csv_writer.writerow(row)
			i += 1
	if active:
		# save the output file
		output_file.flush()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', required=True, help="A directory containing files with both _cnt and _mrk type EEG files. No trailing slash please.")
	parser.add_argument('-m', '--mode', required=False, default='train', help="Set to 'test' if parsing test (evaluation) files. Otherwise use default 'train' for training (calibration) files.")
	parser.add_argument('-l', '--classLength', required=False, default=3000, help="Length of class. This means: for how many data points was each class label held.")
	opts = parser.parse_args()
	return opts

def main():
	# get directory of data files
	opts = parse_args()
	input_dir = opts.dir

	# create target dir if needed
	output_dir = input_dir + "/bcikit_parsed"
	if not os.path.exists(output_dir):
		print "output dir did not exists, creating", output_dir
		os.makedirs(output_dir)

	# loop through all files in source dir
	os.chdir(input_dir)
	for fname in glob.glob("*_cnt*"):
		output_fname = fname.replace("_cnt", "_bcikit_parsed")
		print "---------------------------------------------"
		print fname, " > ", output_fname
		output_filepath = os.path.join(output_dir,output_fname)
		print "mode:",opts.mode
		parseFile(fname, output_filepath, opts.mode, opts.classLength)

	print "FINISHED"
	exit()

if __name__ == "__main__":
	main()

