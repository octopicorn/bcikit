__author__ = 'odrulea'

import argparse
from lib.devices import RABBITMQ_ADDRESS, MOCK_DEVICE_ID, get_supported_metrics, get_supported_devices
from Analysis.AnalysisService import AnalysisService
from Viz.VisualizationServer import VisualizationServer
from multiprocessing import Process

_SUPPORTED_DEVICES = get_supported_devices()
_SUPPORTED_METRICS = get_supported_metrics()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--device_id', required=True,
                        help="A unique ID to identify the device you are sending data from. "
                             "For example: 'octopicorn2015'")
    parser.add_argument('-d', '--device_name', required=True,
                        help="The name of the device your are sending data from. "
                             "Supported devices are: %s" % _SUPPORTED_DEVICES)
    parser.add_argument('-q', '--mq_host', default="localhost",
                        help="The address of the RabbitMQ message queue you are sending data to.\n"
                             "Use " + RABBITMQ_ADDRESS + " to send data to our hosted service. \n"
                                                         "Otherwise use 'localhost' if running CloudBrain locally")
    parser.add_argument('-c', '--conf_path', default="./conf/conf.yml",
                        help="Path to your configuration .yml file (relative to the Analysis directory).\n"
                             "Default is ./conf.yml")

    parser.add_argument('-v', '--viz', default=1, help="Boolean flag: whether or not to run visualization server.\n"
                                                       "0 = do not run server. (Default = 1)")

    opts = parser.parse_args()

    return opts

def main():
    opts = parse_args()

    device_name = opts.device_name
    device_id = opts.device_id
    cloudbrain_address = opts.mq_host
    conf_path = opts.conf_path
    viz = opts.viz


    run(device_name,
        device_id,
        cloudbrain_address,
        conf_path,
        viz
        )

def run(device_name='muse',
        device_id=MOCK_DEVICE_ID,
        cloudbrain_address=RABBITMQ_ADDRESS,
        conf_path=None,
        viz=True
        ):

    # start visualization server in a separate subprocess
    # has to be done in a subprocess because server is a blocking, infinite loop
    if viz == 1:
        vizServer = VisualizationServer(debug=True, conf_path=conf_path)
        p1 = Process(target=vizServer.start)
        p1.daemon = True
        p1.start()
    else:
        print "*** vizualization server disabled ***"

    # start analysis processing chain
    service = AnalysisService(
        device_name=device_name,
        device_id=device_id,
        rabbitmq_address=cloudbrain_address,
        conf_path=conf_path

    )
    service.start()




if __name__ == "__main__":
    main()
