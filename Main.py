# -*- coding: utf-8 -*
# 사용된 SSD 300 Model은 pierluigi ferrari의 ssd_keras (https://github.com/pierluigiferrari/ssd_keras)를 참조하였음.


from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import SSD300_Predict_module as SSD300

app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def test():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    response = SSD300.vision(img)
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

app.run(host="0.0.0.0", port=5000)