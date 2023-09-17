import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
from tflite_runtime.interpreter import Interpreter
import picar_4wd as fc
import random

_MARGIN = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red
_SPEED = 1

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=24):
        # Initialize the camera and image
        self.stream = cv2.VideoCapture(0)
        _ = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        _ = self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,resolution[0])
        _ = self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,resolution[1])

        self.frame = None
        self.stopped = False

    def start(self):
	    # Start the thread for reading frames
      Thread(target=self.update,args=()).start()
      return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (success, self.frame) = self.stream.read()
            if not success:
                self.stream.release()
                sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def run(model: str, labels, res_width: int, res_height: int, min_threshold: float):
    # Initialize interpreter
    interpreter = Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    stream = VideoStream(resolution=(res_width,res_height),framerate=30).start()
    time.sleep(1)

    fc.forward(_SPEED)


    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        
        distance = fc.get_distance_at(0)
        
        frame = stream.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(rgb_frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Create a TensorImage and perform the object detection
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        stop_sign = False
        for i in range(len(scores)):
            if ((scores[i] > min_threshold) and labels[int(classes[i])] == 'stop sign'):
                # Draw box
                ymin = int(max(1,(boxes[i][0] * res_height)))
                xmin = int(max(1,(boxes[i][1] * res_width)))
                ymax = int(min(res_height,(boxes[i][2] * res_height)))
                xmax = int(min(res_width,(boxes[i][3] * res_width)))
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), _TEXT_COLOR, 3)

                # Draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i])) 
                labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.7, 2) #
                label_ymin = max(ymin, labelSize[1] + 10) 
                cv2.putText(frame, label, (xmin, label_ymin-_MARGIN), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
                stop_sign = True
                

        print(distance)
        if stop_sign:
            print('STOP')
            fc.stop()
            time.sleep(2)

            print("Move back and turn right or left")
            # Move back and turn right
            fc.backward(_SPEED)
            time.sleep(.3)
            #randomize the turn side
            turn = random.randint(0,1)
            if turn == 0:
                fc.turn_right(_SPEED*2)
            else:
                fc.turn_left(_SPEED*2)
            time.sleep(2)
            # Stop and restart loop to go forward
            fc.stop()
        else:
            fc.forward(_SPEED)

       # Draw framerate
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_PLAIN,_FONT_SIZE,_TEXT_COLOR,2,_FONT_THICKNESS)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', frame)

    cv2.destroyAllWindows()
    stream.stop()
    fc.stop()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='./detect.tflite')
  parser.add_argument(
      '--labelMap',
      help='Path of the label map.',
      required=False,
      default='./labelmap.txt')
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--minThreshold',
      help='Minimum confidence threshold.',
      required=False,
      type=float,
      default=0.5)
  args = parser.parse_args()

  with open(args.labelMap, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

  run(args.model, labels, args.frameWidth, args.frameHeight, args.minThreshold)

if __name__ == '__main__':
  main()