import pika
import json
from cloudbrain.subscribers.SubscriberInterface import Subscriber
from cloudbrain.utils.metadata_info import get_metrics_names
from cloudbrain.settings import RABBITMQ_ADDRESS


class PikaSubscriber(Subscriber):

  def __init__(self, device_name, device_id, rabbitmq_address, metrics):
    super(PikaSubscriber, self).__init__(device_name, device_id, rabbitmq_address)
    self.connection = None
    self.channel = None
    self.queues = {}
    self.metrics = metrics


  def connect(self):
    credentials = pika.PlainCredentials('cloudbrain', 'cloudbrain')
    self.connection = pika.BlockingConnection(pika.ConnectionParameters(
      host=self.host, credentials=credentials))
    self.channel = self.connection.channel()

    for metric_id,metric_info in self.metrics.iteritems():
        # each metric has an internal id, and a name
        #
        # the internal id is used as a consumer_tag, or think of it as a hardcoded local variable
        # for example, if i have 2 inputs [class_label, data], the 2 inputs are not interchangeable,
        # and we must know which is which, so we use their ids to identify them
        #
        # the name is used to find the correct exchange in rabbitmq
        key = "%s:%s:%s" %(self.device_id,self.device_name, metric_info['name'])
        # declare the exchange serving this metric
        self.channel.exchange_declare(exchange=key, exchange_type='direct')

        # declare queue and bind
        # exclusive=True is important to make a queue that will be destroyed when client hangs up
        # otherwise the queue would persist to the next session, with some old data still stuck in it
        #
        # Another thing, someone might look at this queue declaration and wonder: why didn't we pass in
        # the queue name as a parameter like everywhere else?  The reason for this is that, by not passing in
        # a name, we allow pika to use a randomly generated name for the queue
        self.queues[metric_id] = self.channel.queue_declare(exclusive=True).method.queue
        self.channel.queue_bind(exchange=key, queue=self.queues[metric_id], routing_key=key)

        print "[Subscriber Started] Queue --> [" + key + "] for input [" + metric_id + "]"


  def disconnect(self):
    self.connection.close_file()


  def consume_messages(self, callback):
    # loop through all queues connected and consume all
    for queue_id,queue_name in self.queues.iteritems():
        self.channel.basic_consume(callback,
                      queue=queue_name,
                      exclusive=True,
                      no_ack=True,
                      consumer_tag=queue_id)

    self.channel.start_consuming()


