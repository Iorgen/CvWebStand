import os
import base64
import io
import logging
import json
import aiohttp
import aiohttp_jinja2
import cv2
import numpy as np
from PIL import Image
from aiohttp import web
from faker import Faker
# from core.inference.multi_estimator import DriverMonitoring
from core.HardHat_detection_yolov5 import Hardhat_detection_yolov5
from core.hardhat_classifier_fastai import Clasp_glasses_classifier
from core.utils.general import plot_one_box

import sys
sys.path.insert(0, '../core/')


log = logging.getLogger(__name__)


hardhat_detector = Hardhat_detection_yolov5(view=False, draw=True)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# driver_monitor = DriverMonitoring(None)

# Load classifier
clasp_glasses_classifier = Clasp_glasses_classifier()


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def get_random_name():
    fake = Faker()
    return fake.name()


async def index(request):
    # define connection
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('index.html', request, {})

    # # For preparation
    await ws_current.prepare(request)

    # Random names
    name = get_random_name()
    log.info('%s joined.', name)

    # Send via current websocket success connection
    await ws_current.send_json({'action': 'connect', 'name': name})

    # For each websocket send message that new user connect
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'join', 'name': name})

    # Add current websocket into list of all websockets
    request.app['websockets'][name] = ws_current

    # Main loop
    while True:
        log.info('pre-receive')
        # Wait for get some info from websocket
        msg = await ws_current.receive()
        log.info(msg.type)
        # Define masg is BINARY
        if msg.type == aiohttp.WSMsgType.BINARY:
            # BINARY is an image
            # Inference with preparration
            image = Image.open(io.BytesIO(msg.data))
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
            # frame_result, landmarks, headPosition, output, gaze_vector = driver_monitor.run_per_image(opencvImage)
            # ret, jpeg = cv2.imencode('.jpg', frame_result)
            # print(headPosition)
            # Send back image in bytes
            # await ws_current.send_bytes(data=jpeg.tobytes())
            await ws_current.send_json({'action': 'cv_result', 'landmarks': 'landmark'})
        else:
            break

    # at closing delete ffrom WS list
    del request.app['websockets'][name]
    log.info('%s disconnected.', name)
    # Send to all users that websocket closed
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'disconnect', 'name': name})
    # Return
    return ws_current


async def rtsp_detection_stream(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('rtsp_template.html', request, {})

    await ws_current.prepare(request)
    await ws_current.send_json({'action': 'connect'})

    vcap = cv2.VideoCapture("/Users/jurgen/PycharmProjects/hardhat_aiohttp/data/demo.mp4")
    # vcap = cv2.VideoCapture("rtsp://admin:admin@90.188.118.248:554")

    while (True):
        ret, frame = vcap.read()
        if ret:
            # Inference part DETECTOR
            bboxes_xyx2y2, labels, frame_processed = hardhat_detector.predict(frame)

            # Inference part CLASSIFIER
            clasps_list = []
            glasses_list = []
            for num in range(len(bboxes_xyx2y2)):
                bbox = bboxes_xyx2y2[num]
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                crop = frame[ymin:ymax, xmin:xmax]
                clasp, glasses = clasp_glasses_classifier.predict(crop)
                clasps_list.append(clasp)
                glasses_list.append(glasses)


                # Draw
                label = labels[num]
                text = f'{label}_{clasp}_{glasses}'
                plot_one_box(bbox, frame, label=text, color=[53, 165, 181], line_thickness=3)

            ret, jpeg = cv2.imencode('.jpg', frame_processed)
            # Send back image in bytes
            await ws_current.send_bytes(data=jpeg.tobytes())

    # at closing delete from WS list
    log.info('user disconnected.')
    return ws_current
