#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server.py
# gunicorn -w 10 -b localhost:5000 peaknet.datasets.remote_dataset_server:app

from flask import Flask, request, Response
import os
import io
import numpy as np
from peaknet.datasets.utils_psana import PsanaImg
from peaknet.perf import Timer

app = Flask(__name__)

# Buffer for each process (if using multiple processes with something like Gunicorn)
psana_img_buffer = {}

# Get the current process ID
pid = os.getpid()

def get_psana_img(exp, run, access_mode, detector_name):
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]

@app.route('/fetch-psana', methods=['POST'])
def fetch_psana():
    exp           = request.json.get('exp')
    run           = request.json.get('run')
    access_mode   = request.json.get('access_mode')
    detector_name = request.json.get('detector_name')
    event         = request.json.get('event')
    mode          = request.json.get('mode', 'image')

    psana_img = get_psana_img(exp, run, access_mode, detector_name)

    with Timer(tag = None, is_on = True) as t:
        data = psana_img.get(event, None, mode)
    t_get_image = t.duration

    # Serialize data using BytesIO and npy...
    with Timer(tag = None, is_on = True) as t:
        with io.BytesIO() as buffer:
            np.save(buffer, data)
            buffer.seek(0)
            response_data = buffer.getvalue()
    t_pack_image = t.duration

    print(f"Processed exp={exp}, run={run:04d}, event={event:06d}; psana={t_get_image * 1e3:.2f} ms, packing={t_pack_image * 1e3:.2f} ms.")

    return Response(response_data, mimetype='application/octet-stream')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
