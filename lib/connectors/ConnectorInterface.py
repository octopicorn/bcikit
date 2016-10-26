from abc import ABCMeta, abstractmethod

from lib.devices import get_metrics_names

class Connector(object):
  
    __metaclass__ = ABCMeta
  
    def __init__(self, publishers, buffer_size, step_size, device_name, device_port, device_mac=None):

      self.metrics = get_metrics_names(device_name)
      self.device = None
      self.device_port = device_port
      self.device_name = device_name
      self.device_mac = device_mac

      self.buffers = {metric: ConnectorBuffer(buffer_size, step_size, publishers[metric].publish) for metric in self.metrics}
      self.publishers = publishers

      
    @abstractmethod
    def connect_device(self):
      """
      
      :return:
      """
    
class ConnectorBuffer(object):

  def __init__(self, buffer_size, step_size, callback):
    self.buffer_size = buffer_size
    self.step_size = step_size
    self.callback = callback
    self.message_buffer = []

    self.count = 0

  def write(self, datum):
    """
    add one data point to the buffer

    :param datum:
    :return:
    """
    self.message_buffer.append(datum)
    self.count += 1

    # print(len(self.message_buffer), self.count, self.step_size, self.buffer_size)

    # if len(self.message_buffer) % self.buffer_size == 0:
    if self.count >= self.step_size and len(self.message_buffer) >= self.buffer_size:
      self.callback(self.message_buffer[-self.buffer_size:])
      self.message_buffer = self.message_buffer[-self.buffer_size:]
      self.count = 0
      
