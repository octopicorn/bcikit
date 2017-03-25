/**
 OpenBCI_NodeJS

Copyright © 2017 OpenBCI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
/**
 *
 * main idea here borrowed from OpenBCI github (MIT license shared above)
 *
 * original connector code at:
 * https://github.com/OpenBCI/OpenBCI_NodeJS/blob/master/examples/getStreamingDaisy/getStreamingDaisy.js
 *
 *
 * To install:
 *   install Node
 *   install npm
 *   change to root directory of your project
 *   run `npm install`
 *
 * To run:
 *  `node connect.js`
 *
 *
 * Note: if you get a Node related error like:
 *
 * Error: The module '/path/to/node_modules/serialport/build/Release/serialport.node'
 * was compiled against a different Node.js version using NODE_MODULE_VERSION 48.
 * This version of Node.js requires NODE_MODULE_VERSION 51. Please try re-compiling or re-installing
 * the module (for instance, using `npm rebuild` or`npm install`)
 *
 * Solution is to run this:
 * `npm rebuild --build-from-source`
 *
 *
 *  Parameters: defaults are hard-coded, any params passed in override
 *              name/value parameters: boardType, device_id, device_name
 *              boolean parameters, don't need a value, just presence = true (verbose, log, simulate)
 *
 * Example usage:
 *
 * daisy 16 channel board
 * node connect.js --boardType=daisy --device_id=octopicorn --device_name=openbci16 --verbose
 *
 * daisy 16 channel board, log data to console output as it comes
 * node connect.js --boardType=daisy --device_id=octopicorn --device_name=openbci16 --verbose --log
 *
 * daisy 16 channel board, mock data
 * node connect.js --boardType=daisy --device_id=octopicorn --device_name=openbci16 --verbose --simulate
 *
 * default 8 channel board
 * node tools/connect.js --device_id=octopicorn --device_name=openbci16 --verbose
 *
 * default 8 channel board, log data to console output as it comes
 * node tools/connect.js --device_id=octopicorn --device_name=openbci16 --verbose --log
 *
 * default 8 channel board, mock data
 * node tools/connect.js --device_id=octopicorn --device_name=openbci16 --verbose --simulate
 *
 */
var amqp = require('amqplib/callback_api');
var OpenBCIBoard = require('openbci').OpenBCIBoard;

let MQConnection, MQChannel, MQMessage;

// used to name message queue exchange, as "device_id:device_name:metric_name"
const metric_name = 'eeg';

//var boardType = 'default',
//    debug = false, // Pretty print any bytes in and out... it's amazing...
//    verbose = true, // Adds verbosity to functions
//    hardSet = false,
//    simulate = false;

var args = require('minimist')(process.argv.slice(2));

const {device_id, device_name} = args;
// set defaults, and override from any matches in args
let {
  /* boardType
    default - 8 Channel OpenBCI board (Default)
    daisy - 8 Channel board with Daisy Module - 16 Channels
    ganglion - 4 Channel board
  */
  boardType = 'default',
  debug = false,
  verbose = false,  // info messages
  simulate = false, // fake data
  log = false,     // print data to console
  bufferLength = 10
} = args;

// special setting for daisy
let hardSet = (boardType === 'daisy');

// instantiate OpenBCI board
var ourBoard = new OpenBCIBoard({
  boardType,
  debug,
  hardSet,
  verbose,
  simulate
});

ourBoard.on('error', (err) => {
  console.log(err);
});

ourBoard.autoFindOpenBCIBoard().then(portName => {
  if (portName) {
    /**
     * Connect to the board with portName
     * Only works if one board is plugged in
     * i.e. ourBoard.connect(portName).....
     */
    ourBoard.connect(portName) // Port name is a serial port name, see `.listPorts()`
      .then(() => {
        ourBoard.once('ready', () => {

          // connect to message queue (RabbitMQ)
          amqp.connect('amqp://localhost', function(err, conn) {

            MQConnection = conn;

            // create channel
            MQChannel = conn.createChannel(function(err, ch) {
              var exchange = [device_id, device_name, metric_name].join(':');
              var messageBuffer = [];

              ch.assertExchange(exchange, 'direct', {durable: false});

              // begin streaming from OpenBCI board
              ourBoard.streamStart();
              // when each sample comes in from the board
              ourBoard.on('sample', (sample) => {

                // build message from sample
                MQMessage = {};
                for (let i = 0; i < ourBoard.numberOfChannels(); i++) {
                  MQMessage["channel_"+i] = 10**6 * sample.channelData[i].toFixed(8);
                }
                MQMessage['timestamp'] = new Date().getTime();// milliseconds since start of current epoch

                // add message to message buffer
                messageBuffer.push(MQMessage);

                // if buffer is full
                if(messageBuffer.length === bufferLength){
                  // send message
                  ch.publish(exchange, exchange, new Buffer(JSON.stringify(messageBuffer)));

                  // empty buffer
                  messageBuffer = [];
                }


                // print data to ocmmand line if asked for
                if(log){
                  console.log(MQMessage);
                  console.log(typeof MQMessage)
                }
              });
            });
          });
        });
      });
  } else {
    /** Unable to auto find OpenBCI board */
    console.log('Unable to auto find OpenBCI board');
  }
});

function exitHandler (options, err) {

  if (options.cleanup) {
    if (verbose) console.log('clean');
    ourBoard.removeAllListeners();
    /** Do additional clean up here */
  }
  if (err) console.log(err.stack);
  if (options.exit) {
    if (verbose) console.log('exit');
    ourBoard.disconnect().catch(console.log);
  }
}

if (process.platform === 'win32') {
  const rl = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
  });

  rl.on('SIGINT', function () {
    process.emit('SIGINT');
  });
}

// do something when app is closing
process.on('exit', exitHandler.bind(null, {
  cleanup: true
}));

// catches ctrl+c event
process.on('SIGINT', exitHandler.bind(null, {
  exit: true
}));

// catches uncaught exceptions
process.on('uncaughtException', exitHandler.bind(null, {
  exit: true
}));