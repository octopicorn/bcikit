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
        # Web server root dir
        self.wwwroot = os.path.abspath(os.path.dirname(__file__)) + "/www/"

    def start(self, httpPort = 9999):
        # Create multiplexer
        router = MultiplexConnection.get(ann=ConnectionPlot, bob=ConnectionClassLabel)

        # get config from yaml file
        settings_file_path = os.path.realpath( os.path.join( os.getcwdu(), self.conf_path) )
        stream = file(settings_file_path, 'r')
        # set conf on all multiplexer channels from the yaml config file
        router.set_conf(yaml.load(stream))

        # Register multiplexer
        EchoRouter = SockJSRouter(router, '/echo')

        # Create application
        app = tornado.web.Application(
                EchoRouter.urls +
                [
                    # special handler for index.html, because it's default and you don't actually type index.html
                    (r"/", VisualizationServer.IndexHandler),
                    # html files are treated as templates
                    (r"(?:([^:/?#]+):)?(?://([^/?#]*))?([^?#]*\.(?:html))?", VisualizationServer.MainHandler),
                    # all other static files just go direct to www folder
                    (r"/(.*)", tornado.web.StaticFileHandler, {"path": self.wwwroot})
                ],
                static_path=self.wwwroot,
                template_path=self.wwwroot,
                debug=True
        )
        app.listen(httpPort)

        if self.debug:
            print self.LOGNAME + "Started - all web requests routed to " + self.wwwroot
            print self.LOGNAME + "Visit visualization server by pointing your web browser to http://localhost:" + str(httpPort) + "/index.html"

        # start the tornado server
        tornado.ioloop.IOLoop.instance().start()



    class MainHandler(tornado.web.RequestHandler):
        """
        """
        def get_demo_scripts(self):
            return {
                "eeg_canvas_multi":{
                    "library_link":"http://canvasjs.com/html5-javascript-dynamic-chart/",
                    "title":"EEG Demo with CanvasJS (multi-chart)",
                    "scripts":[
                        "/lib/canvasjs/jquery.canvasjs.min.js",
                        "/demos/eeg/canvasjs/eeg-canvasjs-multi.js"
                    ]
                },
                "eeg_canvas_single":{
                        "library_link":"http://canvasjs.com/html5-javascript-dynamic-chart/",
                        "title":"EEG Demo with CanvasJS (single)",
                        "scripts":[
                            "/lib/canvasjs/jquery.canvasjs.min.js",
                            "/demos/eeg/canvasjs/eeg-canvasjs-single.js"
                        ]
                },
                "eeg_flot":{
                        "library_link":"http://www.flotcharts.org/flot/examples/",
                        "title":"EEG Demo with Flot",
                        "scripts":[
                            "/lib/flot/jquery.flot.js",
                            "/lib/flot/jquery.flot.canvas.js",
                            #"/lib/flot/jquery.flot.downsample.js",
                            "/demos/eeg/flot/eeg-flot.js"
                        ]
                },
                "eeg_epochjs":{
                    "library_link":"https://github.com/epochjs/epoch",
                    "title":"EEG Demo with Epoch",
                    "scripts":[
                        "/lib/canvasjs/jquery.canvasjs.min.js",
                        "/demos/eeg/canvasjs/eeg-canvasjs-multi.js"
                    ],
                    "stylesheets":[
                        ""
                    ]
                },
            }
        def get_demo_script(self, key):
            return self.get_demo_scripts()[key] if key is not None and key in self.get_demo_scripts().keys() else None

        def get(self,a,b,uri):
            #print "MAIN HANDLER"

            script_key = self.get_argument("script",None)
            script_info = self.get_demo_script(script_key)
            self.render(str(self.get_template_path()) + str(uri), script_info=script_info)


    # Index page handler
    class IndexHandler(tornado.web.RequestHandler):
        """Regular HTTP handler to serve the chatroom page"""
        def get(self):
            #print "INDEX HANDLER"
            self.render('index.html')



if __name__ == "__main__":
    server = VisualizationServer()
    server.start()