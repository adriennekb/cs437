import argparse
import time
import sys
import cv2
from threading import Thread
from tflite_runtime.interpreter import Interpreter
import picar_4wd as fc
import numpy as np
import math

# Import necessary functions and classes from your existing files
from astar import astar, getStart, construct_path, getNeighbors, manhattan
from object_detection import VideoStream, run
from object_location import update_map, object_locations

# speed of vehicle
_SPEED = 20

# grid size for object location mapping
_GRID_SIZE = 30

# Initialize array map
array_map = np.zeros((_GRID_SIZE, _GRID_SIZE))
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def setup_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='detect.tflite', help='Path of the object detection model.')
    parser.add_argument('--labelMap', default='labelmap.txt', help='Path of the label map.')
    parser.add_argument('--frameWidth', type=int, default=640, help='Width of frame to capture from camera.')
    parser.add_argument('--frameHeight', type=int, default=480, help='Height of frame to capture from camera.')
    parser.add_argument('--minThreshold', type=float, default=0.5, help='Minimum confidence threshold for object detection.')
    
    return parser

def object_detection(interpreter, input_details, output_details, labels, min_threshold, stream, width, height):
    """Detect objects using the object detection model and return the detections."""

    # Get the current frame from the video stream
    frame = stream.read()

    # Run the current frame through the object detection model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(rgb_frame, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detections = {'stop sign': False, 'traffic cone': False}

    # Check the detections and set the corresponding flags
    for i in range(len(scores)):
        if scores[i] > min_threshold:
            object_label = labels[int(classes[i])]
            if object_label in detections:
                detections[object_label] = True

    return detections, frame

def handle_detections(detections):
    """Handle detections and take appropriate actions based on the detected objects."""

    # Take appropriate actions based on the detected objects
    if detections['stop sign']:
        print('STOP Sign Detected')
        fc.stop()
        time.sleep(2)
    elif detections['traffic cone']:
        print('Traffic Cone Detected')
        fc.stop()
        time.sleep(0.5)
        perform_maneuver()
    else:
        fc.forward(_SPEED)

def perform_maneuver():
    """Perform a maneuver to avoid an obstacle."""
    
    # Execute a maneuver to go around the obstacle
    fc.turn_left(_SPEED)  # Start with a left turn
    time.sleep(1)  # Adjust the time to suit the turning radius
    fc.forward(_SPEED)  # Move forward to go around the obstacle
    time.sleep(1)  # Adjust the time based on the distance to be covered
    fc.turn_right(_SPEED)  # Turn right to realign with the original path
    time.sleep(1)  # Adjust the time to suit the turning radius
    fc.forward(_SPEED)  # Continue moving forward

def steer_car_to_follow_path(path):
    if path:
        for point in path[1:]:
            x, y = point
            #Move 1, if the point is x-1, y, then



def main():
    # Set up argument parser and parse command-line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Initialize video stream with the specified resolution
    video_stream = VideoStream(resolution=(args.frameWidth, args.frameHeight), framerate=30).start()
    time.sleep(1)  # Allow some time for the video stream to start

    # Initialize object detection interpreter
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Load labels for object detection
    with open(args.labelMap, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize A* grid and start location
    global array_map

    args = parser.parse_args()
    goal = (0,0)

    #Get Input and Output Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    while True:
        # Step 1: Update the map with the latest sensor data
        update_map()

        # Step 2: Get the updated grid
        grid = array_map.copy()

        # Step 3: Find the start position (modify getStart function to work with your obstacle data)
        start = getStart(grid)

        # Step 4: Run the A* algorithm to find the optimal path
        path = astar(grid)

        # Step 5: Object detection to identify specific objects (like stop signs) and take appropriate actions
        
        detections, frame = object_detection(interpreter, input_details, output_details, labels, args.minThreshold, video_stream, width, height)
        handle_detections(detections)
        # Overlay the path on the current frame (assuming `path` is a list of (x, y) coordinates)
        for (x, y) in path:
            cv2.circle(frame, (x * 20, y * 20), 5, (0, 255, 0), -1)  # Adjust the multiplier for x and y based on your grid size
    
        # Display the current frame with the path overlay
        cv2.imshow('Self-driving car', frame)
    
        # Return True if the user pressed the 'q' key, indicating they want to quit
        return cv2.waitKey(1) & 0xFF == ord('q')

        # Step 6: Use the path found by A* to guide the car
        steer_car_to_follow_path(path)
        

    # ... (add code to integrate A*, object detection, and object location updates into a loop)
    
if __name__ == "__main__":
    main()
