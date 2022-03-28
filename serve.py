from urllib import response
from flask import Flask, request, send_file
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import json

app = Flask('myApp')

classes = {
0:"INFO_END_OF_EXP_WAY",
1:"PROHIBITORY_NO_JAYWALKING",
2:"INFO_TP_CAMERA_ZONE",
3:"PROHIBITORY_SPD_LIMIT_90",
4:"MANDATORY_TURN_LEFT",
5:"WARNING_ERP",
6:"PROHIBITORY_SPD_LIMIT_70",
7:"INFO_U_TURN_LANE",
8:"MANDATORY_SPLIT_WAY",
9:"MANDATORY_STOP",
10:"PROHIBITORY_SPD_LIMIT_50",
11:"WARNING_CURVE_RIGHT_ALIGNMENT_MARKER",
12:"INFO_ZEBRA_CROSSING",
13:"INFO_RAIN_SHELTER",
14:"PROHIBITORY_NO_ENTRY",
15:"MANDATORY_KEEP_LEFT",
16:"INFO_PARKING_AREA_FOR_MOTORCARS",
17:"INFO_PEDESTRIAN_USE_CROSSING",
18:"WARNING_RESTRICTED_ZONE_AHEAD",
19:"WARNING_CURVE_LEFT_ALIGNMENT_MARKER",
20:"INFO_START_OF_EXP_WAY",
21:"MANDATORY_GIVE_WAY",
22:"PROHIBITORY_NO_VEH_OVER_HEIGHT_4.5",
23:"PROHIBITORY_SPD_LIMIT_40",
24:"WARNING_SLOW_SPEED",
25:"WARNING_ROAD_HUMP",
26:"PROHIBITORY_NO_LEFT_TURN",
27:"INFO_ONE_WAY_RIGHT",
28:"INFO_ONE_WAY_LEFT",
29:"WARNING_SLOW_DOWN",
30:"WARNING_MERGE",
31:"PROHIBITORY_NO_RIGHT_TURN"
}

trf_model = load_model('./saved_models/final/final-sgtrafficloc/sg_traffic_loc.h5')
img_ht = 50
img_wd = 50
channels = 3

@app.route('/')
def home():
    return {"success": True}, 200

@app.route('/predict', methods = ['POST'])
def make_predictions():
    user_input = request.get_json(force=True)
    
    data_pred = np.array(user_input)

    loc_pred = trf_model.predict(data_pred)

    classes_xtest = np.argmax(loc_pred[0], axis=1)

    pred_class = classes[classes_xtest[0]]

    return {'response': [pred_class, int(loc_pred[1][0][0] * img_ht), int(loc_pred[1][0][1] * img_wd), int(loc_pred[1][0][2] * img_ht), int(loc_pred[1][0][3] * img_wd)]}


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=int(os.environ.get("PORT", 8080)))