# bcikit
Modular analysis of biosensor data streams, with visualization component, especially designed for EEG signals.
Currently only supports OpenBCI hardware.

## What it does
Primary objective is a modular processing chain, which can analyze incoming data streaming from an EEG device.
Signal processing tasks include:
- Mock signal generation for testing (random, sine wave)
- Class label generation (these can also be thought of as "cues" for the BCI user)
- Notch filter
- Bandpass filter (high, low)
- FFT (frequency analysis)
- DWT (frequency-time-phase analyis)
- Signal conversion (from scalar to cartesian coordinates for plotting)
- Fixed length windowing (on a rolling basis, with overlap, for training phase)
- Class segregated windowing (for testing phase, and online use)
- Downsampling (using simple decimation, or advanced LTTB algorithm)
- Machine Learning (realtime data with scikit-learn)

## Limitations
This software is designed with the open-source user in mind. In the neurotech community, this type of use case is 
 often designated "consumer-grade", as distinguished from "research-grade".  This distinction serves to set expectations
 about the resources and expertise of the user.  For example, a home user operating a $500 device, with no formal 
 neuroscience background, will be expected to have a different set of requirements than an academic user, in a lab
 setting, operating a $10,000+ system in the service of funded research.
 
In general, "consumer grade" means a device offering lower spatial resolution (1-16 electrodes, vs 32-128 in higher end 
systems), and lower sampling rate (100-250Hz, vs 1000Hz), which means more modest BCI challenges can be targeted. Some
examples of "consumer-grade" projects include: motor imagery detection, neurofeedback training, P300 speller, SSVEP
controlled applications, and general entertainment projects, like controlling remote control toys and vehicles.

## Prerequisites

Python 2.7
(note: the excellent Anaconda scientific package for Python is highly recommended to satisfy the most commonly used
dependencies quickly) 

- numpy (install with conda or pip)
- scikit-learn (install with conda or pip)
- json (install with conda or pip)
- tornado http://www.tornadoweb.org
- cloudbrain https://github.com/marionleborgne/cloudbrain (note: you'll need to install CloudBrain's dependencies too, 
such as liblo.  You can skip the Cassandra install step, unless you're really sure you will be using it.)
 
## Installation
Install using normal procedure

Choice A: you plan to modify the code yourself (likely)
```
python setup.py develop
```
Choice B: you plan to only use this as third-party tool
```
python setup.py install
```

## Quickstart: Run Analysis and Visualization in One Call
Sometimes, you might prefer to start/stop the AnalysisService.py and VisualizationServer.py in their own separate 
terminal window, and this is supported.  However, for convenience, most people will want to start/stop both at once.
You can start both processes by a script at root dir called "run.py"
```
python {path-to-bcikit}/run.py  -i octopicorn -d openbci -c conf/viz_demos/eeg.yml
```

Parameters to run.py are as follows:

| short        | verbose           | required  | description    |
| :----------- |:------------------|:----------|:---------------|
|-i|--device_id|Yes|A unique ID to identify the device you are sending data from. For example: 'octopicorn2015'.|
|-d|--device_name|Yes|The name of the device your are sending data from. Supported devices are: openbci, muse. Must be used even if you're using SignalGenerator with no actual device connected (device specs are needed even for mock data).|
|-q|--mq_host|No (default=**localhost**)|The address of the RabbitMQ message queue you are sending data to. Use 'localhost' if running locally, or cloudbrain.rocks for CloudBrain's hosted service.|
|-c|--conf_path|No (default=**conf.yml**)|Path to your configuration .yml file (relative to the root bcikit/ directory)|

In the future, run.py will likely be the preferred method to run bcikit, since visualization and analysis will share 
certain startup variables and conf file.



## Quickstart Alternative: Run CloudBrain independently of AnalysisModules and VisualizationServer (2 terminal windows) 
* if you plan to use a device connector from CloudBrain, you can start that up first to begin streaming data
(mock connector example given)
```
  python {cloudbrain path}/cloudbrain/publishers/sensor_publisher.py --mock -n openbci -i octopicorn -c localhost -p 99
```
* Start both Analysis and Visualization by a single convenience script at root dir called "run.py" (should be started
with the same device id, name, rabbitmq params as above)
```
  python {bcikit path}/run.py -i octopicorn -c localhost -n openbci
```
* Point your web browser to http://localhost:9999/index.html - Currently the eeg/flot is the only demo working
* Ctrl-C to stop analysis process & viz server
* Edit analysis processing chain and parameters by modifying AnalysisModules/conf.yml You can try commenting out
module blocks to turn them on or off. Set debug to True to see live output from command line.

## Analysis Modules Overview
In the folder "AnalysisModules", find the conf.yml.  This file is used to set a processing chain of analysis modules,
defining the order, names of input and output metrics, and any special params used by each module.

For now, the defined analysis modules include:
- ModuleSignalGenerator (generate mock data, either random or sine wave) 
- ModuleWindows (collect raw data into rolling windows/matrices of fixed size, with optional overlap)
- ModuleClassWindows (collect raw data + class labels into rolling windows/matrices of variable size, grouped by class label, no overlap)
- ModuleConvert (conversions, example: convert matrix of raw data to coordinate pairs (x,y) - used for plotting) 
- ModuleDownsample (decrease number of points while still retaining essential features of graph, used for plotting only)
- ModuleTest (used as a template for new modules)

It is up to the user to make sure that, if one module follows another, that module 1 output is compatible with module 2 
input.  For example, if a module is expecting scalar input, don't have its input connected to output from a module that 
emits matrices.

Some limitations to be aware of:
- only works with openbci device type for now
- the config vars **inputs > data > message_queues** and **inputs > data > message_queues** define metric names to read/write on rabbitmq
- only rabbitmq is supported, not pipes
- there are some random things hardcoded

## Analysis Modules Demo
* start streaming data using cloudbrain with a mock openbci (assumes you're running rabbitmq locally)
```
python {cloudbrain path}/cloudbrain/publishers/sensor_publisher.py --mock -n openbci -i octopicorn -c localhost -p 99
```
* start the analysis modules script in a separate terminal window, using same device id and rabbitmq host
```
python {CloudbrainAnalysis path}/AnalysisModules/AnalysisService.py -i octopicorn -c localhost -n openbci
```
If debug is on for a given module, it should output to command line.


## Visualizations Overview
This is very rough.  Under the folder "Viz", there are demos, intended to show a chart visualization per metric/module, per library.
For example, starting with raw eeg, there will be an example using libraries flot, chart.js v1.0, chart.js v2.0,
rickshaw, and other libraries.

The intent here is to provide for many different impementations to be demoed so we can compare performance and
feasibility of different chart libraries for a specific type of visualization.

For example, in general, we prefer chart libraries that support WebGL.  However, the main library we've tried that
supports this, chart.js, only offers 5 types of chart.  So, it makes sense to branch out and use different libraries for
different visualizations, rather than try and find one lib that works well for everything.

The visualizations run using a tornado server copying the pattern used in cloudbrain's rt_server and frontend.
The server was modified to use the "multiplex" capability, so that we are not limited to one connection per window. This
was modelled after a sockjs example found here:
https://github.com/mrjoes/sockjs-tornado/tree/master/examples/multiplex
That is why there is reference to "ann" and "bob".  The server is defined in **Viz/VisualizationServer.py**.

The tornado server just passes through all get requests to their relative location based on the "www" folder as root.
That is why all the visualization code is presently stuffed into the folder "Multiplex".

If you establish an "ann" type connection, it will use a PlotConnection, which is just a proxy for the connection
defined in connection_subscriber.py. This is meant for output from cloudbrain to a javascript visualization via websocket.

If you establish a "bob" type connection, it will use a ClassLabelConnection, which, as the type name suggests, is
meant for the javascript frontend to actually send data back to cludbrain via websocket.  This is not yet implemented.
The idea here is that the frontend visualization will require the capability to show some UI to the user meant for
training (or calibration) sessions, in which capturing class label tag is critical.

The current limitations are so many it's not worth listing out.  Only outputs of type (message_type: "MATRIX", 
data_type: "RAW_COORDS") are supported in any of the visualization demos, so that means the only outputs that can be 
charted must come from either ModuleConvert or ModuleDownsample.  

Different libraries perform differently in different browsers.  Overall, Canvas.js library has (so far) shown the best 
performance, as seen here: [http://jsperf.com/amcharts-vs-highcharts-vs-canvasjs/16].  For optimal performance, we 
recommend the latest Chrome browser + Canvas.js.  Although Canvas.js was shown to have excellent performance in 
static drawing benchmarks in Safari (at the above link), in real-world tests of dynamic drawing (i.e. live streaming), 
Safari is very choppy and laggy.

Pro-tip: In Chrome you can turn on an FPS clock, which will help to measure real performance and visualize GPU memory 
usage. Go to chrome://flags and turn on **FPS counter**

## Visualizations Demo
1. assuming you're running a mock connector as specified in step 1 above, you can start your server by
```
python {CloudbrainAnalysis path}/Viz/VisualizationServer.py
```

2. open your browser and visit the path you want, relative to the www folder, like this
http://localhost:9999/index.html
(only working demo for now is the flot eeg)

3. the "eeg" metric should work if you have mock connector streaming.  The basic idea is that you pick the metric you
want to see and click "connect" to start streaming it.  The actual websocket connection is opened when the page loads.


## Data Representation
For calculations on EEG data, we will follow the machine learning conventions of "vector" and "matrix".  For anyone
new to this field, it's helpful to know that a *vector* is simply an array. And a *matrix* is an array of arrays.

Specifically, in EEG data, all of the data coming from a single electrode is considered an array of voltage values over
time, or a *vector* containing time-series data for that electrode.  Each electrode, or "channel", is represented as a
vector of voltage readings.

When you have more than one electrode, you now have multiple vectors, and we will put the vectors together in a matrix.

Suppose that, for an EEG system with output resolution of 250Hz, you have 3 electrodes, and you're analyzing one second
of data from this system.  Since our system is 250Hz, or 250 readings per second, this means we have 250 readings x
3 channels.  By using a matrix, this will be represented as a grid. We have two choices:

##### Example 1
##### 250 x 3 matrix 
|time |0  |1  |2  |3  |4  |5  |6  |7  |8  |9  |...|
|-----|---|---|---|---|---|---|---|---|---|---|---|
|Fz   |0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1.0|...|
|C3   |0.0|0.2|0.4|0.6|0.8|1.0|1.2|1.4|1.6|1.8|...|
|C4   |0.3|0.6|0.9|1.2|1.5|1.8|2.1|2.4|2.7|3.0|...|


##### Example 2 
##### 3 x 250 matrix
|time |Fz | C3| C4|
|-----|---|---|---|
|0    |0.1|0.0|0.3|
|1    |0.2|0.2|0.6|
|2    |0.3|0.4|0.9|
|3    |0.4|0.6|1.2|
|4    |0.5|0.8|1.5|
|5    |0.6|1.0|1.8|
|6    |0.7|1.2|2.1|
|7    |0.8|1.4|2.4|
|8    |0.9|1.6|2.7|
|9    |1.0|1.8|3.0|
|...  |...|...|...|


The choice of whether to represent electrode channel data vectors as rows (example 1), or as columns (example 2), is 
debatable.  Our convention in CloudbrainAnalysis will be to use a 250 columns and 3 rows (example 1).  Each row will 
represent all the datapoints of a single electrode channel.  Each column will represent data across all channels at one 
point in time.

One justification for this is simply that this is intuitively how we tend to think of time series data, with the
horizontal axis of a graph meaning "time".  Additionally, this is a similar convention to what is used in scikit-learn
and MNE packages.  

It has been suggested that there is performance optimization which can be realized by performing calculations on 
elements which are contiguous in memory.  You can read more about that in the section "Note on array order" here: 
http://scikit-image.org/docs/dev/user_guide/numpy_images.html

## Matrix Operations
Here's a quick primer on numpy's matrix for those who might be new to python's way of handling matrices.

The np.matrix() object can be thought of as a wrapper around the basic np.array() (type='ndarray').  
A matrix is just an array of arrays:
```
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

By using the matrix object, and not just ndarray, we get access to some useful matrix math functions.  However, just 
keep in mind that the actual data is contained internally in a np.array().  You can always access this by the A property
of the matrix.  Or, if you just want to print string, just print the matrix directly.
```
window = np.matrix(np.random.rand(3,6))
foo = window.A
print foo
print window
```

Once you have a matrix, you can access specific channels and slices of data using matrix notation. This may be 
familiar to anyone who has used Matlab.

```
# matrix of random values
# 3 EEG channels (rows)
# 6 timepoints (columns)
window = np.matrix(np.random.rand(3,6))
print window
#
#[[ 0.01252493  0.76089514  0.90413342  0.14933877  0.95271887  0.62169743]
# [ 0.4461164   0.67827997  0.47488861  0.66459204  0.96701774  0.65374514]
# [ 0.22725775  0.35366613  0.04000928  0.362111    0.49086496  0.04899759]]
#
print "vector representing all channels' data at timepoint 5:"
print window[:,4]
#
#[[ 0.95271887]
# [ 0.96701774]
# [ 0.49086496]]
#
print "vector representing only channel 2 data for the entire time series in the window:"
print window[1,:]
#
#[[ 0.4461164   0.67827997  0.47488861  0.66459204  0.96701774  0.65374514]]
#
print "vector representing only channel 2 data between timepoint 3 and 5 (notice that range includes starting point 3, but not endpoint 5, so by saying [between 3 and 5], you're getting [3,4]):"
print window[1,2:4]
#
#[[ 0.47488861  0.66459204]]
#
print "vector representing only channel 2 data from timepoint 3 to the end (including starting point 3):"
print window[1,2:]
#
#[[ 0.47488861  0.66459204  0.96701774  0.65374514]]
#
print "vector representing only channel 2 data from beginning to timepoint 3 (not including endpoint 3):"
print window[1,:2]
#
#[[ 0.4461164   0.67827997]]
#
```

### To Do
- establish a convention for modules to specify what kinds of visualization they are compatible with.
- establish a convention whereby, if any module in configuration has specified a visualization component, the
visualization server will be auto-started

### Metrics To Be Implemented
- Bandpass Filter (High, Low)
- Common Spatial Pattern (CSP)
- Discrete Wavelet Transform (DWT)
- Notch Filter (60 Hz, etc)
- Noise Removal
- Eyeblink and EMG Artifact removal
- Channel Visibility (i.e. 8 channels coming in, 2 channels coming out)



