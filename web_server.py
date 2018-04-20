#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import os.path
import tornado.escape
from tornado import gen
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import json
import deyzm


g_yzm_classfier = deyzm.load_classfier()

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", HomeHandler),
            (r"/deyzm", DeyzmHandler)
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
            cookie_secret="603550C2-DB60-4518-BC98-819F0A206DCC",
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)

        print('I am listening now...')


class BaseHandler(tornado.web.RequestHandler):
    pass

class HomeHandler(BaseHandler):
    def get(self):
        self.write('Home')

class DeyzmHandler(BaseHandler):
    def get(self):
        self.write('use post')
    
    def post(self):
        p = self.get_argument('p')
        ret = deyzm.yzm_predict(g_yzm_classfier, p)
        print('ret', ret)
        self.write(ret)

def main():
    define("port", default=3001, help="run on the given port", type=int)

    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
