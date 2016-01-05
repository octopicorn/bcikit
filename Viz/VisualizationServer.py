# -*- coding: utf-8 -*-

import tornado.ioloop
import tornado.web
from sockjs.tornado import SockJSRouter

from Viz.Server.Multiplex import MultiplexConnection
from Viz.Server.Connections import ConnectionPlot, ConnectionClassLabel
import os
import yaml

class VisualizationServer:

    LOGNAME = "[Visualization Server] "

    def __init__(self, debug=False, conf_path='/conf.yml'):
        self.debug = debug
        self.conf_path = conf_path
        self.conf = None

    def start(self, httpPort = 9999):
        # Create multiplexer
        router = MultiplexConnection.get(ann=ConnectionPlot, bob=ConnectionClassLabel)

        # get config from yaml file
        location = os.path.realpath( os.path.join(os.getcwdu(), "AnalysisModules" ) )
        settings_file_path = os.path.join(location, self.conf_path)
        stream = file(settings_file_path, 'r')
        # set conf on all multiplexer channels from the yaml config file
        router.set_conf(yaml.load(stream))

        # Register multiplexer
        EchoRouter = SockJSRouter(router, '/echo')

        # Web server root dir
        wwwroot = os.path.abspath(os.path.dirname(__file__)) + "/www"

        # Create application
        app = tornado.web.Application(
                EchoRouter.urls +
                [
                    # special handler for index.html, because it's default and you don't actually type index.html
                    (r"/", VisualizationServer.IndexHandler),
                    # all other static files just go direct to www folder
                    (r"/(.*)", tornado.web.StaticFileHandler, {"path": wwwroot})
                ]
        )
        app.listen(httpPort)

        if self.debug:
            print self.LOGNAME + "Started - all web requests routed to " + wwwroot
            print self.LOGNAME + "Visit visualization server by pointing your web browser to http://localhost:" + str(httpPort) + "/index.html"

        # start the tornado server
        tornado.ioloop.IOLoop.instance().start()



    # Index page handler
    class IndexHandler(tornado.web.RequestHandler):
        """Regular HTTP handler to serve the chatroom page"""
        def get(self):
            self.render('www/index.html')

if __name__ == "__main__":
    server = VisualizationServer()
    server.start()