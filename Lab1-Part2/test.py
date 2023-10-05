import picar_4wd as fc
import time
import random
import numpy as np
import sys
import math

# speed of vehicle
speed = 20

#fill a 2D array with 0s that is 100x100(grid_size)
grid_size = 30
array_map = np.zeros((grid_size, grid_size))
# set the numpy printing size for screen reading and text files
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def update_map():
    # set car at bottom center of the array
    car_location = (grid_size-1, math.floor(grid_size/2)-1)

    # set starting angle at -90
    angle = -90
    fc.get_distance_at(angle)
    # wait for the servo to reach destination to avoid misreadings
    time.sleep(0.3)

    # marking the car as 2 to distinguish location on array
    array_map[car_location] = 2
    # print(array_map)
    while angle <= 90:
        distance = fc.get_distance_at(angle)
        # this will only print distances of objects within the grid size set
        if 0 < distance < grid_size:
            print(angle, '° ->', distance, "cm")
        else:
            print(angle, '° ->', '/')
        # this will try to put a number 1 if it detects anything within 
        # the grid size
        if 0 < distance < grid_size:
            try:
                a = (distance*(math.cos(math.radians(angle))))
                b = (math.sqrt(distance**2-a**2))
                if angle < 0:
                    b *= -1
                a = (grid_size-1) - a
                b = (math.floor(grid_size/2)-1) - b
                # will map only if the object is within the grid size
                if 0 < a < grid_size and 0 < b <grid_size:
                    array_map[math.floor(a), math.floor(b)] = 1
                    # We are assuming objects are within 1 cm so we add
                    # extra 1 cm to both left and right to account for 
                    # the 5 degrees that are not tested and fill the 0 gaps
                    try:
                        array_map[math.floor(a), math.floor(b-1)] = 1
                        array_map[math.floor(a), math.floor(b+1)] = 1
                    except:
                        print("something broke mapping it", a, b)
            except:
                print("something broke in math function", a, b)
        angle += 5
    # print(array_map)
    fc.get_distance_at(0)

    # Saving the array in a text file to see results
    file = open("map.txt", "w+")
    content = str(array_map)
    file.write(content)
    file.close()

# this function is used just for testing, enable it in main
def object_locations():
    for y, x in np.ndindex(array_map.shape):
        if array_map[x, y] == 1:
            print("Object at location: " + str(y) + ", " + str(x))
        elif array_map[x, y] == 2:
            print("Car at location: " + str(y) + ", " + str(x))

def test():
    fc.forward(speed)
    while True:
        # Find distance, then turn right if close
        distance = fc.get_distance_at(0)
        print(distance)
        if distance < 12.0:
            print("Move back and turn right or left")
            # Move back and turn right
            fc.backward(speed)
            time.sleep(.3)
            #randomize the turn side
            turn = random.randint(0,1)
            if turn == 0:
                fc.turn_right(speed*2)
            else:
                fc.turn_left(speed*2)
            time.sleep(2)
            # Stop and restart loop to go forward
            fc.stop()
        else:
            fc.forward(speed)
        time.sleep(.1)
        
if __name__ == '__main__':
    try:
        update_map()
        # object_locations()
        # test()
    except KeyboardInterrupt:
        fc.stop()
        exit(0)