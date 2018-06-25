# -*- coding: utf-8 -*
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

import cv2
import random
import json
import tensorflow as tf

img_height = 270
img_width = 480
n_classes = 9

# 1: Build the Keras model
K.clear_session() # 메모리에서 이전 모델 초기화
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.
weights_path = 'SSD300_trained/SSD300_dataset_180529_2_weights.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
graph = tf.get_default_graph()

def vision(source):
    orig_images = [] # store the images here.
    input_images = [] # store resized versions of the images here.
    orig_images.append(source)
    # 이미지 변환
    img = cv2.resize(source, (img_width,img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    # ML구간 비동기 처리를 위한 부분
    global graph
    with graph.as_default():
        y_pred = model.predict(input_images)

        confidence_threshold = 0.5

        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        classes = ['background',
                   'paprika', 'egg', 'orange', 'apple',
                   'tomato', 'coke', 'pepsi', 'Beans',
                   'dressing']

        boxed = orig_images[0]

        return_data = {}
        labellist = []
        for box in y_pred_thresh[0]:
            if box[1] >= 0.6:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = int(box[2] * orig_images[0].shape[1] / img_width)
                ymin = int(box[3] * orig_images[0].shape[0] / img_height)
                xmax = int(box[4] * orig_images[0].shape[1] / img_width)
                ymax = int(box[5] * orig_images[0].shape[0] / img_height)
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                color1 = random.randint(0, 255)
                color2 = random.randint(0, 255)
                color3 = random.randint(0, 255)
                cv2.putText(boxed, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color1,color2,color3), 1, cv2.LINE_AA)
                cv2.rectangle(boxed, (xmin, ymin), (xmax, ymax), (color1,color2,color3), 2)
                name = classes[int(box[0])]
                prob = '{:.2f}'.format(box[1])
                # JSON 생성 부분
                value = [name, prob, xmin, xmax, ymin, ymax]
                labellist.append(value)
        cv2.imwrite('result_SSD300/{}.png'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), orig_images[0])
        return_data['predict'] = labellist
        json_data = json.dumps(return_data)
        del(return_data)
        del(labellist)
        return json_data