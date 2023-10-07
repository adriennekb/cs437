document.onkeydown = updateKey;
document.onkeyup = resetKey;

var server_port = 65432;
var server_addr = "10.0.0.222";   // the IP address of your Raspberry PI
var fc = 0;

const dataToSend = {
    fc: 0,
    input: ""
};

function client(fc){
    
    const net = require('net');
    var input = document.getElementById("message").value;

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        // send the message
        dataToSend.input = input;
        dataToSend.fc = fc;
        client.setEncoding('utf-8')
        const jsonData = JSON.stringify(dataToSend);
        client.write(jsonData);
        // client.write(`${input}\r\n`);           
    });

    // get the data from the server
    client.on('data', (data) => {
        let r_data = JSON.parse(data);
        document.getElementById("speed").innerHTML = r_data.speed;
        document.getElementById("cpu_temperature").innerHTML = r_data.cpu_temperature;
        document.getElementById("gpu_temperature").innerHTML = r_data.gpu_temperature;
        document.getElementById("cpu_usage").innerHTML = r_data.cpu_usage;
        document.getElementById("battery").innerHTML = r_data.battery;
        document.getElementById("raw").innerHTML = data;
        console.log(data.toString());
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });


}

// for detecting which key is been pressed w,a,s,d
function updateKey(e) {
    e = e || window.event;

    if (e.keyCode == '87') {
        console.log("key pressed!!!")
        // up (w)
        document.getElementById("upArrow").style.color = "green";
        // send_data("87");
        fc = 87;
        console.log("fc: " + fc)
    }
    else if (e.keyCode == '83') {
        // down (s)
        document.getElementById("downArrow").style.color = "green";
        // send_data("83");
        fc = 83;
    }
    else if (e.keyCode == '65') {
        // left (a)
        document.getElementById("leftArrow").style.color = "green";
        // send_data("65");
        fc = 65;
    }
    else if (e.keyCode == '68') {
        // right (d)
        document.getElementById("rightArrow").style.color = "green";
        // send_data("68");
        fc = 68;
    }
}

// reset the key to the start state 
function resetKey(e) {

    e = e || window.event;
    fc = 0;

    document.getElementById("upArrow").style.color = "grey";
    document.getElementById("downArrow").style.color = "grey";
    document.getElementById("leftArrow").style.color = "grey";
    document.getElementById("rightArrow").style.color = "grey";
}


// update data for every 50ms
function update_data(){
    setInterval(function(){
        // get image from python server
        client(fc);
    }, 50);
}
