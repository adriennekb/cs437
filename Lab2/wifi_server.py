# This file goes in your raspberrypi

import socket
import os
import subprocess
import json
import picar_4wd as fc
from picar_4wd.utils import pi_read

HOST = "10.0.0.222" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)
speed = 20

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    try:
        while 1:
            client, clientInfo = s.accept()
            print("server recv from: ", clientInfo)
            data = client.recv(1024)      # receive 1024 Bytes of message in binary format
            d_data = data.decode()
            print(d_data)
            if data != b"":
                print(data)
            if "87" in d_data:
                print("forwards")
                fc.forward(speed)
            elif "83" in d_data:
                print("backwards")
                fc.backward(speed)
            elif "65" in d_data:
                print("turn left")
                fc.turn_left(speed)
            elif "68" in d_data:
                print("turn right")
                fc.turn_right(speed)
            else:
                fc.stop()
            temp_dict = pi_read()
            temp_dict["speed"] = speed
            json_string = json.dumps(temp_dict)
            client.sendall(json_string.encode('utf-8')) # Echo back to client
            
    except: 
        print("Closing socket")
        client.close()
        s.close()    