RABBITMQ_ADDRESS = 'localhost'

MOCK_DEVICE_ID = "mock"

DEVICE_METADATA = [
  {'device_name': 'openbci',
   'device_type': 'eeg_headset',
   'metrics':
     [
       {
         'metric_name': 'eeg',
         'num_channels': 8,
         'metric_description': 'raw eeg data from OpenBCI 8 channel'
       }
     ]
  },
  {'device_name': 'openbci16',
   'device_type': 'eeg_headset',
   'metrics':
     [
       {
         'metric_name': 'eeg',
         'num_channels': 16,
         'metric_description': 'raw eeg data from OpenBCI 16 channel'
       }
     ]
  },
  {
    'device_name': 'muse',
    'device_type': 'eeg_headset',
    'metrics':
      [
        {
          'metric_name': 'eeg',
          'num_channels': 4,
          'metric_description': 'Raw eeg data coming from the 4 channels of the Muse'
        },
        {
          'metric_name': 'horseshoe',
          'num_channels': 4,
          'metric_description': 'Status indicator for each channel (1 = good, 2 = ok, >=3 bad)'
        },
        {
          'metric_name': 'concentration',
          'num_channels': 1,
          'metric_description': None
        },
        {
          'metric_name': 'mellow',
          'num_channels': 1,
          'metric_description': None
        },
        {
          'metric_name': 'acc',
          'num_channels': 3,
          'metric_description': None
        },
        {
          'metric_name': 'delta_absolute',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'theta_absolute',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'beta_absolute',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'alpha_absolute',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'gamma_absolute',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'delta_relative',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'theta_relative',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'beta_relative',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'alpha_relative',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'gamma_relative',
          'num_channels': 4,
          'metric_description': None
        },
        {
          'metric_name': 'is_good',
          'num_channels': 4,
          'metric_description': 'Strict data quality indicator for each channel, 0= bad, 1 = good.'
        },
        {
          'metric_name': 'blink',
          'num_channels': 1,
          'metric_description': None
        },
        {
          'metric_name': 'jaw_clench',
          'num_channels': 1,
          'metric_description': None
        },

      ]
  },
  {
    'device_name': 'neurosky',
    'device_type': 'eeg_headset',
    'metrics': [
      {
        'metric_name': 'concentration',
        'num_channels': 1,
        'metric_description': None
      },
      {
        'metric_name': 'meditation',
        'num_channels': 1,
        'metric_description': None
      },
      {
        'metric_name': 'signal_strength',
        'num_channels': 1,
        'metric_description': None
      },
    ]
  }
]

class _DeviceNameNotFound(Exception):
  pass


def map_metric_name_to_num_channels(device_name):
  """
  Map wearable metric names to the number of channels of this metric.
  :return: dict {metric_name: num_channels}
  """
  metadata = [metadata for metadata in DEVICE_METADATA if metadata['device_name'] == device_name]

  if len(metadata) > 0:
    metrics = metadata[0]['metrics']
  else:
    raise _DeviceNameNotFound("Could not find device name '%s' in metadata" % device_name)

  metric_name_to_num_channels = {}
  for metric in metrics:
    metric_name_to_num_channels[metric['metric_name']] = metric['num_channels']

  return metric_name_to_num_channels


def get_metrics_names(device_type):
  """
  Get metric names for a specific device type.
  :return: list of metric names
  """
  metadata = [metadata for metadata in DEVICE_METADATA if metadata['device_name'] == device_type]

  if len(metadata) > 0:
    metrics = metadata[0]['metrics']
  else:
    raise _DeviceNameNotFound("Could not find device name '%s' in metadata" % device_type)

  metric_names = []
  for metric in metrics:
    metric_names.append(metric['metric_name'])

  return metric_names


def get_supported_devices():
  return [device['device_name'] for device in DEVICE_METADATA]


def get_supported_metrics():
  metrics = []
  for device_name in get_supported_devices():
    metrics.extend(get_metrics_names(device_name))
  return metrics

def get_num_channels(device_name, metric):
  metric_to_channels = map_metric_name_to_num_channels(device_name)
  return metric_to_channels[metric]
