__author__ = 'odrulea'
# -*- coding: utf-8 -*-
"""
I just had to make some changes to the cloudbrain rt_server in order to be able to use it.
Here is a list of all the changes I've made:

1. this rt_server is not really a server anymore, it's just one connection, of which multiple could be used
2. the host for RABBITMQ_ADDRESS should not be hard-coded to go to cloudbrain.rocks. I've kept that as the default,
but added a settable property on the connection configuration JSON passed from the html if someone wants to use
localhost, for example
OD

"""
import json
import logging

from sockjs.tornado.conn import SockJSConnection
from lib.devices import RABBITMQ_ADDRESS

from TornadoSubscriber import TornadoSubscriber
from lib.utils import BufferToMatrix, ListConfOutputMetrics
from lib.constants import *


#logging.getLogger().setLevel(logging.ERROR)

class ConnectionPlot(SockJSConnection):
    """RtStreamConnection connection implementation"""
    # Class level variable
    clients = set()
    conf = None

    def __init__(self, session):
        super(self.__class__, self).__init__(session)
        self.subscribers = {}

    def send_probe_factory(self, metric_name, data_type=None):

        def send_probe(body):
            #logging.debug("GOT [" + metric_name + "]: " + body)
            #print "GOT [" + metric_name + "]: [" + data_type + "]"
            if data_type == MESSAGE_TYPE_MATRIX:
                """
                MESSAGE_TYPE_MATRIX output is a base64 encoded blob which needs to be decoded
                """
                buffer_content = json.loads(body)
                for record in buffer_content:
                    message = BufferToMatrix(record, output_type="list")
                    self.send(message,True)
            else:
                """
                MESSAGE_TYPE_TIME_SAMPLE output is a dict object, with one element for each channel + one for timestamp
                gotcha: Pika tends to make all keys in the dict utf8
                """

                buffer_content = json.loads(body)
                for record in buffer_content:
                    record["metric"] = metric_name
                    self.send(json.dumps(record))

        return send_probe


    def on_open(self, info):
        logging.info("Got a new connection...")
        # debug
        print "[Tornado Server: ConnectionPlot] opened websocket connection"
        self.clients.add(self)
        metrics = ListConfOutputMetrics(self.conf, prefix="viz_")
        #print metrics

        # set a json payload to deliver to UI as a "handshake"
        menu = {"type":"handshake","metrics":metrics}

        # add extra info to the handshake as needed
        global_conf = self.conf['global'] if 'global' in self.conf else {}
        if global_conf:
            if 'num_channels' in global_conf:
                menu['num_channels'] = global_conf['num_channels']


        # send the handshake to the clients

        self.send(json.dumps(menu))


    def on_message(self, message):
        """
        This will receive instructions from the client to change the
        stream. After the connection is established we expect to receive a JSON
        with deviceName, deviceId, metric; then we subscribe to RabbitMQ and
        start streaming the data.
        """
        msg_dict = json.loads(message)
        if msg_dict['type'] == 'subscription':
            self.handle_channel_subscription(msg_dict)
        elif msg_dict['type'] == 'unsubscription':
            self.handle_channel_unsubscription(msg_dict)
        elif msg_dict['type'] == 'command':
            self.handle_channel_command(msg_dict)

    def handle_channel_subscription(self, stream_configuration):
        # parameters that can be passed in JSON from client
        device_name = (stream_configuration['deviceName'] if "deviceName" in stream_configuration else None)
        device_id = (stream_configuration['deviceId'] if "deviceId" in stream_configuration else None)
        metric = (stream_configuration['metric'] if "metric" in stream_configuration else None)
        data_type = (stream_configuration['dataType'] if "dataType" in stream_configuration else None)
        rabbitmq_address = (stream_configuration['rabbitmq_address'] if 'rabbitmq_address' in stream_configuration else RABBITMQ_ADDRESS)

        # debug
        print "[Tornado Server] Received SUBSCRIBE for Queue [" + device_id + ":" + device_name + ":" + metric + "]"

        if metric not in self.subscribers:
            self.subscribers[metric] = TornadoSubscriber(callback=self.send_probe_factory(metric, data_type),
                                       device_name=device_name,
                                       device_id=device_id,
                                       rabbitmq_address=rabbitmq_address,
                                       metric_name=metric)

        self.subscribers[metric].connect()

    def handle_channel_unsubscription(self, unsubscription_msg):
        # parameters that can be passed in JSON from client
        metric = (unsubscription_msg['metric'] if 'metric' in unsubscription_msg else '')

        # debug
        print "[Tornado Server] Received UNSUBSCRIBE for metric [" + metric + "]"

        if unsubscription_msg['metric'] in self.subscribers:
            self.subscribers[unsubscription_msg['metric']].disconnect()

    def handle_channel_command(self, stream_configuration):
        # parameters that can be passed in JSON from client
        command = (stream_configuration['command'] if "command" in stream_configuration else None)
        device_name = (stream_configuration['deviceName'] if "deviceName" in stream_configuration else None)
        device_id = (stream_configuration['deviceId'] if "deviceId" in stream_configuration else None)
        metric = (stream_configuration['metric'] if "metric" in stream_configuration else None)
        data_type = (stream_configuration['dataType'] if "dataType" in stream_configuration else None)
        rabbitmq_address = (stream_configuration['rabbitmq_address'] if 'rabbitmq_address' in stream_configuration else RABBITMQ_ADDRESS)

        # debug
        print "[Tornado Server] Received SUBSCRIBE for Queue [" + device_id + ":" + device_name + ":" + metric + "]"

        if metric in self.subscribers:
            self.subscribers[metric].command(command)

    def on_close(self):
        logging.info('Disconnecting client...')
        for metric in self.subscribers.keys():
            subscriber = self.subscribers[metric]
            if subscriber is not None:
                logging.info('Disconnecting subscriber for metric: ' + metric)
                subscriber.disconnect()

        self.subscribers = {}
        #self.timeout.stop()
        self.clients.remove(self)
        logging.info('Client disconnection complete!')

    def send_heartbeat(self):
        self.broadcast(self.clients, 'message')

"""
This connection type is meant to handle class label tags coming from the frontend UI, and routes them back to server
"""
class ConnectionClassLabel(SockJSConnection):
    """
    Connection to collect class labels
    """
    conf = None

    def on_open(self, info):
        self.send('ClassLabel Connection established: server will begin receiving data.')

    def on_message(self, message):
        self.send('ClassLabel Connection got your message: ' + message)