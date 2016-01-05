__author__ = 'odrulea'

import base64
import json
import numpy as np


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

def ListConfOutputMetrics(conf):
    """
    'modules': [
        {
            'outputs': {
                'data': {
                    'buffer_size': 0, 'message_queues': 'eeg', 'message_type': 'TIME_SAMPLE', 'data_type': 'RAW_DATA'
                }
            },
            'class': 'ModuleSignalGenerator', 'id': 'foo1', 'settings': {'range': [-40, 40], 'num_channels': 16, 'frequency': 10, 'sampling_rate': 250, 'debug': False, 'pattern': 'sine'}}, {'inputs': {'data': {'message_type': 'TIME_SAMPLE', 'name': 'eeg', 'data_type': 'RAW_DATA'}}, 'class': 'ModuleWindows', 'id': 'foo4', 'outputs': {'data': {'buffer_size': 0, 'message_queues': 'viz_window', 'message_type': 'MATRIX', 'data_type': 'RAW_DATA'}}, 'settings': {'debug': False, 'samples_per_window': 21, 'num_channels': 16, 'window_overlap': 0}}, {'inputs': {'data': {'message_type': 'MATRIX', 'name': 'viz_window', 'data_type': 'RAW_DATA'}}, 'class': 'ModuleConvert', 'id': 'foo6', 'outputs': {'data': {'buffer_size': 0, 'message_queues': 'viz_eeg', 'message_type': 'MATRIX', 'data_type': 'RAW_COORDS'}}, 'settings': {'debug': False, 'num_channels': 16}}]}
    """
    metrics = []
    for module in conf['modules']:
        if 'outputs' in module and 'data' in module['outputs']:
            if 'message_queues' in module['outputs']['data']:
                if type(module['outputs']['data']['message_queues']) == str:
                    metrics.append(module['outputs']['data']['message_queues'])
                elif type(module['outputs']['data']['message_queues']) == list:
                    for metric in module['outputs']['data']['message_queues']:
                        metrics.append(metric)

    return metrics