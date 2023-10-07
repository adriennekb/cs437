# This file goes in your raspberrypi

import socket
import os
import subprocess
import json
import picar_4wd as fc
from picar_4wd.utils import pi_read

HOST = "192.168.1.53" # IP address of your Raspberry PI
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
            print(data)
            if data != b"":
                print(data)
            if data == b"87":
                fc.forward(speed)
            elif data == b"83":
                fc.backward(speed)
            elif data == b"65":
                fc.turn_left(speed)
            elif data == b"68":
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