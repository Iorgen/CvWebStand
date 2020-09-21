import base64
import io
import logging

import aiohttp
import aiohttp_jinja2
import cv2
import numpy as np
from core.inference.multi_estimator import DriverMonitoring
from PIL import Image
from aiohttp import web
from faker import Faker

log = logging.getLogger(__name__)


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def get_random_name():
    fake = Faker()
    return fake.name()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
driver_monitor = DriverMonitoring(None)


async def index(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('index.html', request, {})

    await ws_current.prepare(request)

    name = get_random_name()
    log.info('%s joined.', name)

    await ws_current.send_json({'action': 'connect', 'name': name})

    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'join', 'name': name})
    request.app['websockets'][name] = ws_current

    while True:
        log.info('pre-receive')
        msg = await ws_current.receive()
        log.info(msg.type)
        # if msg.type == aiohttp.WSMsgType.text:
        #     for ws in request.app['websockets'].values():
        #         if ws is not ws_current:
        #             await ws.send_json(
        #                 {'action': 'sent', 'name': name, 'text': msg.data})
        if msg.type == aiohttp.WSMsgType.BINARY:
            image = Image.open(io.BytesIO(msg.data))
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            opencvImage = driver_monitor.run_per_image(opencvImage)
            #
            # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
            #                                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            # # Draw the rectangle around each face
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(opencvImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.imwrite('some_image.jpg', opencvImage)
            ret, jpeg = cv2.imencode('.jpg', opencvImage)
            await ws_current.send_bytes(data=jpeg.tobytes())
        else:
            break

    del request.app['websockets'][name]
    log.info('%s disconnected.', name)
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'disconnect', 'name': name})

    return ws_current
