import picar_4wd as fc
import time

speed = 20

def test():
    fc.forward(speed)
    while True:
        # Find distance, then turn right if close
        distance = fc.get_distance_at(0)
        print(distance)
        if distance < 12.0:
            print("Move back and turn right")
            # Move back and turn right
            fc.backward(speed)
            time.sleep(.3)
            fc.turn_right(speed*2)
            time.sleep(2)
            # Stop and restart loop to go forward
            fc.stop()
        else:
            fc.forward(speed)
        time.sleep(.1)
        
if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:
        fc.stop()
        exit(0)