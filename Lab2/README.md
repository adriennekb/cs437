Before continuing, make sure you are in the correct directory cs437/Lab2/. "cd Lab2"

For Bluetooth connection
1. Follow Step 1 RPi setup on doc
2. After installing bluedot run the following on RPi
3. bluetoothctl
4. discoverable on
5. pairable on
6. agent on
7. default-agent
8. scan on (look for your computer and copy its mac address just by highlighting it)
9. pair *your pc MAC address* or if that doesn't work, connect through your pc itself Settings->Bluetooth(Might need to enable advanced Bluetooth in it's settings to see RPi)
10. Edit pi_socket.py in your RPi and add it's controller MAC address in the code (you will see the controller MAC address in step 4)
11. Edit windows_socket.py in your PC and add your RPi MAC address in the code
12. Run pi_socket in RPi then windows_socket in PC
13. You should now see the RPi on PC and PC on Rpi counting up

#Electron installation
1. Install nodejs https://nodejs.org/en/download
2. Make sure node and npm are installed "node -v" "npm -v"
3. Initialize configuration with "npm init"
4. Install electron "npm install --save-dev electron"
5. Run app "npm start"