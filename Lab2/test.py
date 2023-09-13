import picar_4wd as fc
import time
import random
import numpy as np
import sys
import math

speed = 20
#fill a 2D array with 0s that is 100x100(grid_size)
grid_size = 30

array_map = np.zeros((grid_size, grid_size))
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def update_map():
    #set car at bottom center of the array
    car_location = (grid_size-1, math.floor(grid_size/2)-1)
    angle = -90
    fc.get_distance_at(angle)
    time.sleep(0.3)
    #marking the car as 2 to distinguish
    array_map[car_location] = 2
    # print(array_map)
    while angle <= 90:
        distance = fc.get_distance_at(angle)
        if 0 < distance < grid_size:
            print(angle, '° =', distance, "cm")
        else:
            print(angle, '° =', '/')
        
        if 0 < distance < grid_size:
            try:
                a = (distance*(math.cos(math.radians(angle))))
                b = (math.sqrt(distance**2-a**2))
                if angle < 0:
                    b *= -1
                a = (grid_size-1) - a
                b = (math.floor(grid_size/2)-1) - b
                if 0 < a < grid_size and 0 < b <grid_size:
                    array_map[math.floor(a), math.floor(b)] = 1
                    try:
                        array_map[math.floor(a), math.floor(b-1)] = 1
                        array_map[math.floor(a), math.floor(b+1)] = 1
                    except:
                        print("something broke mapping it", a, b)
            except:
                print("something broke in math function", a, b)
        angle += 5
    # print(array_map)

    file = open("map.txt", "w+")
    # Saving the array in a text file
    content = str(array_map)
    file.write(content)
    file.close()

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
        object_locations()
        # test()
    except KeyboardInterrupt:
        fc.stop()
        exit(0)