import logging

import jinja2

import aiohttp_jinja2
from aiohttp import web
from aiohttpdemo_chat.views import rtsp_detection_stream, index
from aiohttpdemo_chat.api.views import RecognizeHardHatView


async def init_app():

    app = web.Application()

    app['websockets'] = {}

    app.on_shutdown.append(shutdown)

    aiohttp_jinja2.setup(
        app, loader=jinja2.PackageLoader('aiohttpdemo_chat', 'templates'))

    app.router.add_get('/', index)
    app.router.add_get('/rtsp_stream_test', rtsp_detection_stream)
    app.router.add_view("/api/", RecognizeHardHatView)
    return app


async def shutdown(app):
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


def main():
    logging.basicConfig(level=logging.DEBUG)

    app = init_app()
    web.run_app(app)


if __name__ == '__main__':
    main()
