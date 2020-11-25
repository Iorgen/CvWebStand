import io
from PIL import Image
from aiohttp.web import View
import aiohttp_jinja2
from aiohttp import web
from datetime import datetime


class RecognizeHardHatView(web.View):

    async def get(self):
        return aiohttp_jinja2.render_template('image_form_sample.html', self.request, {})

    async def post(self):
        if self.request.body_exists:
            image = await self.request.read()
            print(type(image))
            image = Image.open(io.BytesIO(image))
        return web.json_response({"room": 1, "token": 1, "result": "OK"})