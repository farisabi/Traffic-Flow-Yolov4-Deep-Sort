import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import csv
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

pts = [deque(maxlen=100) for _ in range(9999)]

# calcLine1 = [ [(200,100), (400,100)], [(200,100), (400,100)] ]
# calcLine2 = [ [(1290,150), (1390,150)], [(1290, 150), (1390, 150)] ]

calcLine1 = [ [(200, 400), (400, 400)], [(200, 400), (400, 400)] ]  #Down [(from(x,y), to(x,y)]]
calcLine2 = [ [(1200, 488), (1321, 500)], [(1200, 488), (1321, 500)] ]  #Down [(from(x,y), to(x,y)]

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
temp_up_t2list = []
temp_down_t2list = []

up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
up_t2list = [0,0,0,0]
down_t2list = [0,0,0,0]

# Function for count vehicle
# def count_vehicle(track, temp_up_list, temp_down_list, up_list, down_list):
#     id = track.track_id
#     index = track.get_class()
#     bbox = track.to_tlbr()
#     iy = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
#
#     # Find the current position of the vehicle
#     if (iy == calcLine1[0]):
#         if id not in temp_up_list:
#             temp_up_list.append(id)
#     elif iy == calcLine2[0]:
#         if id not in temp_down_list:
#             temp_down_list.append(id)
#
#     elif iy == calcLine1[0]:
#         if id in temp_down_list:
#             temp_down_list.remove(id)
#             up_list[index] = up_list[index] + 1
#         elif (index == 'car'):
#             up_list[0] +1
#         elif (index == 'motorbike'):
#             up_list[1] +1
#         elif (index == 'bus'):
#             up_list[2] +1
#         else:
#             up_list[3] +1
#     elif iy == calcLine2[0]:
#         if id in temp_up_list:
#             temp_up_list.remove(id)
#             down_list[index] = down_list[index] += 1
#         elif (index == 'car'):
#             down_list[0] +1
#         elif (index == 'motorbike'):
#             down_list[1] +1
#         elif (index == 'bus'):
#             down_list[2] +1
#         else:
#             down_list[3] +1

def cross_Calculation(i, id, now_label, last_CENTROID, now_CENTROID, temp_up_list, temp_down_list, up_list, down_list):
    calculateLine1 = calcLine1[i]
    calculateLine2 = calcLine2[i]

    if (last_CENTROID[1] < calculateLine1[0][1]) and (now_CENTROID[1] >= calculateLine1[0][1]) and (now_CENTROID[0]>=calculateLine1[0][0]) and (now_CENTROID[0]<=calculateLine1[1][0]):
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif (last_CENTROID[1]<calculateLine2[0][1]) and (now_CENTROID[1]>=calculateLine2[0][1]) and (now_CENTROID[0]>=calculateLine2[0][0]) and (now_CENTROID[0]<=calculateLine2[1][0]):
        if id not in temp_down_list:
            temp_down_list.append(id)


    if (last_CENTROID[1] < calculateLine1[0][1]) and (now_CENTROID[1] >= calculateLine1[0][1]) and (now_CENTROID[0]>=calculateLine1[0][0]) and (now_CENTROID[0]<=calculateLine1[1][0]):
        if id in temp_down_list:
            temp_down_list.remove(id)
            #up_list[now_label] = up_list[now_label] + 1
        elif (now_label == "car"):
            up_list[0] += 1
        elif (now_label == "motorbike"):
            up_list[1] += 1
        elif (now_label == "bus"):
            up_list[2] += 1
        else:
            up_list[3] += 1

    elif (last_CENTROID[1]<calculateLine2[0][1]) and (now_CENTROID[1]>=calculateLine2[0][1]) and (now_CENTROID[0]>=calculateLine2[0][0]) and (now_CENTROID[0]<=calculateLine2[1][0]):
        if id in temp_up_list:
            temp_up_list.remove(id)
            #down_list[now_label] = down_list[now_label] + 1
        elif (now_label == "car"):
            down_list[0] += 1
        elif (now_label == "motorbike"):
            down_list[1] += 1
        elif (now_label == "bus"):
            down_list[2] += 1
        else:
            down_list[3] += 1



def draw_CalculateLine(frame, i):
    calculateLine1 = calcLine1[i]
    calculateLine2 = calcLine2[i]

    cv2.line(frame, (calculateLine1[0][0], calculateLine1[0][1]), (calculateLine1[1][0], calculateLine1[1][1]),
             (0, 0, 155), 2)
    cv2.line(frame, (calculateLine2[0][0], calculateLine2[0][1]), (calculateLine2[1][0], calculateLine2[1][1]),
             (0, 0, 155), 2)

    return frame

def bbox2Centroid(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    return (int(x+(w/2)), int(y+(h/2)))

def printText(img, up_list, down_list):
    y1 = 30
    y2 = 55
    y3 = 80
    y4 = 105

    cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
    cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (360, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (360, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (360, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (360, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    return img




def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None


    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()


        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','bus','truck','motorbike']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        stepa = {}
        for track in tracker.tracks:
            # print("(pre-upd)Tracker.track_id: {},.to_tlwh: {}, Class: {}".format(str(track.track_id), str(track.to_tlwh()), class_name))
            # print("stepa[]:{}".format([track.track_id, track.to_tlwh()]))
            stepa[track.track_id] = track.to_tlwh()

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        draw_CalculateLine(frame, 0)
        draw_CalculateLine(frame, 1)

        for track in tracker.tracks:

            if track.track_id in stepa:
                # print("track existed in prev fr:{}".format([track.track_id,bbox2Centroid(stepa.get(track.track_id)) ] ) )
                # print("cur fr:{}".format([ track.track_id , bbox2Centroid( track.to_tlbr() ) ] ) )
                last_CENTROID = bbox2Centroid(stepa.get(track.track_id))
                now_CENTROID = bbox2Centroid(track.to_tlwh())
                now_label = track.get_class()
                id = track.track_id

                cross_Calculation(0, id, now_label, last_CENTROID, now_CENTROID, temp_up_list, temp_down_list, up_list, down_list)
                cross_Calculation(1, id, now_label, last_CENTROID, now_CENTROID, temp_up_t2list, temp_down_t2list, up_t2list, down_t2list)

            #count_vehicle(track, temp_up_list, temp_down_list, up_list, down_list)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()


            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))

            # track_id[center]
            pts[track.track_id].append(center)

            thickness = 5
            # center point
            cv2.circle(frame, center, 1, color, thickness)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(box_id.track_id),
                                                                                                    class_name, (
                                                                                                    int(bbox[0]),
                                                                                                    int(bbox[1]),
                                                                                                    int(bbox[2]),
                                                                                                    int(bbox[3]))))
            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)

                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)\

        frame = printText(frame, up_list, down_list)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, 'up')
        down_list.insert(0, 'down')
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
