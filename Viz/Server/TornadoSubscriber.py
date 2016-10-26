# -*- coding: utf-8 -*-
"""

"""
import logging
import pika
from tornado.ioloop import IOLoop

#logging.getLogger().setLevel(logging.ERROR)

# Based on: https://pika.readthedocs.org/en/0.9.14/examples/tornado_consumer.html
class TornadoSubscriber(object):

    def __init__(self, callback, device_name, device_id, rabbitmq_address, metric_name):
        self.callback = callback
        self.device_name = device_name
        self.device_id = device_id
        self.metric_name = metric_name

        self.connection = None
        self.channel = None

        self.host = rabbitmq_address
        self.queue_name = None

    def command(self,command):
        print "*********************"
        print "GOT COMMAND:",command
        print "*********************"

    def connect(self):
        credentials = pika.PlainCredentials('cloudbrain', 'cloudbrain')
        self.connection = pika.adapters.tornado_connection.TornadoConnection(pika.ConnectionParameters(
                                        host=self.host, credentials=credentials),
                                        self.on_connected,
                                        stop_ioloop_on_close=False,
                                        custom_ioloop=IOLoop.instance())

    def disconnect(self):
        if self.connection is not None:
            self.connection.close()

    def on_connected(self, connection):
        self.connection = connection
        self.connection.add_on_close_callback(self.on_connection_closed)
        self.connection.add_backpressure_callback(self.on_backpressure_callback)
        self.open_channel()

    def on_connection_closed(self, connection, reply_code, reply_text):
        self.connection = None
        self.channel = None

    def on_backpressure_callback(self, connection):
        logging.info('******** Backpressure detected for ' + self.get_key())
        print '******** Backpressure detected for ' + self.get_key()

    def open_channel(self):
        self.connection.channel(self.on_channel_open)

    def on_channel_open(self, channel):
        self.channel = channel
        self.channel.add_on_close_callback(self.on_channel_closed)
        # self.setup_exchange(self.EXCHANGE)
        # self.channel.confirm_delivery(self.on_delivery_confirmation)
        #logging.info("Declaring exchange: " + self.get_key())
        self.channel.exchange_declare(self.on_exchange_declareok,
                                      exchange=self.get_key(),
                                      type='direct',
                                      passive=True)
        # self.queue_name = self.channel.queue_declare(exclusive=True).method.queue

    def on_channel_closed(self, channel, reply_code, reply_text):
        self.connection.close()

    def on_exchange_declareok(self, unused_frame):
        # exclusive=True is important to make a queue that will be destroyed when client hangs up
        # otherwise the queue would persist to the next session, with some old data still stuck in it
        self.channel.queue_declare(self.on_queue_declareok, queue=self.get_key(), exclusive=True)

    def on_queue_declareok(self, unused_frame):
        #logging.info("Binding queue: " + self.get_key())
        self.channel.queue_bind(
                       self.on_bindok,
                       exchange=self.get_key(),
                       queue=self.get_key(),
                       routing_key=self.get_key())

    def on_bindok(self, unused_frame):
        self.channel.add_on_cancel_callback(self.on_consumer_cancelled)
        self.consumer_tag = self.channel.basic_consume(self.on_message, self.get_key())

    def on_consumer_cancelled(self, method_frame):
        if self.channel:
            self.channel.close()

    def on_message(self, unused_channel, basic_deliver, properties, body):
        self.acknowledge_message(basic_deliver.delivery_tag)
        self.callback(body)

    def acknowledge_message(self, delivery_tag):
        self.channel.basic_ack(delivery_tag)

    def get_key(self):
        key = "%s:%s:%s" %(self.device_id,self.device_name, self.metric_name)
        return key
