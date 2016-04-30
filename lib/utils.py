__author__ = 'odrulea'

import base64
import json
import numpy as np
from scipy.signal import butter as butter
import time
import pandas as pd
import mne
from collections import Counter


DTYPE_COORD = np.dtype([('x', np.float), ('y', np.float)])

def MatrixToBuffer(ndarray, dtype = None):
    """
    In order to get a numpy matrix (array of arrays) into a json serializable form, we have to do a base64 encode
    We will wrap the matrix in an envelope with 3 elements:
    1. type of the ndarray
    2. the entire ndarray encoded as a base64 blob
    3. a list describing the dimensions of the ndarray (2 element list: [rows, cols])

    borrowed from: http://stackoverflow.com/questions/13461945/dumping-2d-python-array-with-json
    :param ndarray:
    :return:
    """
    if dtype is None:
        dtype = ndarray.dtype
    if dtype == DTYPE_COORD:
        dtype = "DTYPE_COORD"
    return [str(dtype),base64.b64encode(ndarray),ndarray.shape]

def BufferToMatrix(jsonDump, output_type=None):
    """
    After retrieving the encoded json from the message queue buffer, we need to translate the 3 element json
    back into its original form.
    The 3 elements are:
    0: use the 0th element to cast correct type
    1: the base64 encoded data just needs to be base64 decoded
    2: use the 2nd element to set correct dimensions with reshape()

    borrowed from: http://stackoverflow.com/questions/13461945/dumping-2d-python-array-with-json
    :param jsonDump:
    :return:
    """
    # if incoming json has not yet been json decoded, do it
    if isinstance(jsonDump, str):
        jsonDump = json.loads(jsonDump)

    # handle type
    dtype_string = jsonDump[0]
    if "DTYPE_" in dtype_string:
        # if the type
        dtype = np.dtype(eval(dtype_string))
    else:
        dtype = np.dtype(dtype_string)

    # reconstitute the data, using cast to decoded type from above
    matrix_array = np.frombuffer(base64.decodestring(jsonDump[1]),dtype)
    if len(jsonDump) > 2:
        matrix_array = matrix_array.reshape(jsonDump[2])

    if output_type is 'list':
        matrix_list = []
        vector = []
        [rows,cols] = matrix_array.shape
        if dtype_string == "DTYPE_COORD":
            for row in np.arange(rows):
                for column in np.arange(cols):
                    point = [matrix_array[row][column]["x"],matrix_array[row][column]["y"]]
                    vector.append(point)
                matrix_list.append(vector)
                vector = []
        else:
            for row in np.arange(rows):
                for column in np.arange(cols):
                    vector.append(matrix_array[row][column])
                matrix_list.append(vector)
                vector = []

        return matrix_list

    return matrix_array

def MatrixToCoords(input, typeSafe=False):
    """
    A matrix of scalar values (i.e. float) is a useful for performing fast calculations on data from many EEG channels.
    However, if you want to plot that data, many charting libraries expect the convention of [x,y] coordinates.
    This function is helpful for performing that conversion.

    Convert an incoming matrix of scalar values to a matrix of [x,y] coordinates
    loop through all points in a matrix and assign [x,y] coordinates based on
    x=row, y=scalar value of original input
    so for example, an input matrix like this:
    [[1 1 0 1]
     [2 1 1 1]
     [1 2 3 1]]

    will become this:
    [[[0, 1] [1, 1] [2, 0] [3, 1]]
     [[0, 2] [1, 1] [2, 1] [3, 1]]
     [[0, 1] [1, 2] [2, 3] [3, 1]]]

    Now any row can be used to plot time series for than channel on an x,y chart.

    Example usage:
    # matrix input
    matrix1 = np.matrix([[1,1,0,1],[2,1,1,1],[1,2,3,1]])
    print matrix1
    print
    foo = MatrixToCoords(matrix1)
    print "Matrix input:"
    print foo

    # numpy array input
    np1 = np.array([[1,1,0,1],[2,1,1,1],[1,2,3,1]])
    foo2 = MatrixToCoords(np1)
    print "Np.array input:"
    print foo2

    # list input
    list1 = [[1,1,0,1],[2,1,1,1],[1,2,3,1]]
    foo3 = MatrixToCoords(list1, True)
    print "List input:"
    print foo3
    """

    # an optional param "typeSafe" can enforce input to be compatible data type, if passing in a raw list
    if(typeSafe):
        if type(input) == list:
            # it's a list, must convert to array
            input = np.array(input)
        if type(input) != np.ndarray and type(input) != np.matrix:
            return None

    # output matrix will be same size as input, but of dtype=list
    # since each value will be a coordinate [x,y]
    (rows,cols) = input.shape
    output = np.ndarray(shape=(rows,cols), dtype=DTYPE_COORD)

    # initalize row and col counters
    row = col = 0
    # loop through matrix by rows, then cols
    while col < cols:
        row=0
        while row < rows:
            # get the y value of the coordinate by looking at same [row,col]
            # location on the input matrix
            output[row,col]["x"] = col
            output[row,col]["y"] = input[row,col]
            row=row+1
        col = col+1

    return output

def ListConfOutputMetrics(conf, prefix=None):
    """
    Loop through all of the available output metrics, among all of the modules defined in the conf .yml file.
    If an option prefix is specified, this will limit response sent to the UI to only include those metrics w/ the prefix
    """
    metrics = []
    for module in conf['modules']:
        if 'outputs' in module and 'data' in module['outputs']:
            if 'message_queues' in module['outputs']['data']:
                if type(module['outputs']['data']['message_queues']) == str:
                    if prefix is None or module['outputs']['data']['message_queues'].startswith(prefix):
                        metrics.append(module['outputs']['data']['message_queues'])
                elif type(module['outputs']['data']['message_queues']) == list:
                    for metric in module['outputs']['data']['message_queues']:
                        if prefix is None or metric.startswith(prefix):
                            metrics.append(metric)

    return metrics

def FilterCoefficients(filter_type, sampling_frequency, boundary_frequencies):
    """
    """
    return butter(2,boundary_frequencies/(sampling_frequency / 2.0), filter_type)

def BCIFileToEpochs(filename=None, num_channels=8, max_epochs_per_class=None, filter_class_labels=[-1,1], epoch_size=50,   include_electrodes=None):
    """

    :param filename:
    :param num_channels:
    :param max_epochs:
    :param filter_class_labels:
    :param epoch_size:
    :param include_electrodes:
    :return:
    """
    print
    print "opening test data file..."
    epochsCounter = Counter()
    window = pd.read_table(filename, header=None, dtype=np.float)

    if include_electrodes is not None:
        raw_data = window.iloc[:,include_electrodes].values.T
        # since we're only using a subset of channels, overwrite the num_channels used to build the result matrix columns
        num_channels, num_samples = raw_data.shape
    else:
        raw_data = window.iloc[:,:num_channels].values.T

    print "channel data", raw_data.shape

    class_labels = window.iloc[:,-1].values
    print "class labels", class_labels.shape, class_labels


    """
    initialize a standard 3-dim array just to have a data structure that is standard with other libs
    In most other libraries, CSP uses a 3 dimensional epoch input: (n_epochs, n_channels, n_times)
    """
    epochs = np.zeros((0,num_channels,epoch_size))
    y = np.array([])

    # find first index of each class switch
    window_start_indexes = np.nonzero(np.r_[1,np.diff(class_labels)[:-1]])
    num_windows = len(window_start_indexes[0])
    #print "there are", num_windows, "class windows"
    #print "window_class_boundaries", window_start_indexes

    for i in xrange(num_windows):
        start_index = window_start_indexes[0][i]
        if start_index != window_start_indexes[0][-1] :
            end_index = window_start_indexes[0][i+1]
        # else:
        #     end_index = None

            this_class = class_labels[start_index]

            if (max_epochs_per_class is None or epochsCounter[this_class] < max_epochs_per_class) and int(this_class) in filter_class_labels:
                #print "next window from ", start_index, "to", end_index
                #print class_labels[start_index:end_index]

                # save the epoch to the standard 3 dim data structure
                nextWindow = np.array(raw_data[:,start_index:end_index])
                num_epochs = np.floor_divide(nextWindow.shape[1], epoch_size)
                #print "will split ",class_labels[start_index],"window of len", nextWindow.shape[1], "into",num_epochs,"epochs of size", epoch_size

                for j in xrange(num_epochs):
                    if max_epochs_per_class is None or epochsCounter[this_class] < max_epochs_per_class:
                        epochsCounter[this_class] += 1
                        y = np.append(y, this_class)
                        start_epoch = start_index + (epoch_size * j)
                        end_epoch = start_epoch + epoch_size
                        nextEpoch = np.array(raw_data[:,start_epoch:end_epoch])
                        epochs = np.append(epochs, [nextEpoch], axis=0)

    print "test file retrieved epochs:", epochs.shape
    print "test file retrieved y:", len(y)
    print epochsCounter

    return [epochs, y]