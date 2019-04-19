var express = require('express')
var app = express()
var http = require('http').Server(app);
var io = require('socket.io')(http)
var pythonShell = require('python-shell');

var pyShells = {}


app.get('/', function(req, res){
    res.sendFile(__dirname + '/main.html')
});

app.use('/static', express.static('static'));

var count = 1;
var roomList = [];


// 연결되는 순간 -> client에서 socket 연 순간
io.on('connection', function(socket){
    console.log('user Connect', socket.id)

    socket.emit('connection', {
        type : 'connected',
        count: count
    });

    socket.on('connection', function(data){
        var isConnected = false;
        roomList.forEach(function(room){
            if(room == data.type+data.room)
                isConnected = true;
        });

        if(isConnected) return false;

        if(data.type == 'join') {
            socket.join(socket.id);
            socket.room = socket.id;
            socket.emit('status', 'wait');

            var options = {
                mode: 'text',
                pythonPath: '/conda/envs/keras/python.exe',
                pythonOptions: ['-u'],
                scriptPath: './',
                args: [socket.id]
            };
            // 실제로는 spark-submit ( sparkApp 을 가동시켜야함. )
            pythonShell.PythonShell.run('ml.py', options, function (err, results) {
                console.log(results);
                if (err) throw err;
            });
        }
        else if(data.type == 'mlModule') {
            socket.join(data.room)
            socket.room = data.room;
            socket.type = data.type;
            console.log(data);
            roomList.push(data.type + data.room);
        }
    });

    /** 연결종료 */
    socket.on('disconnect', function(){
        console.log('user disconnected', socket.id);
        console.log('room disconnected', socket.room);
        roomList.pop(socket.type+socket.room);

        console.log(roomList);
    });

    // 머신러닝 진행상태 -> client
    socket.on('status', function(data){
        console.log(data);
        io.to(socket.room).emit('status', data);
    })
    
    // client -> python 데이터 전송
    socket.on('sendData', function(data){
        console.log(data)
        io.to(socket.room).emit('sendData', data);
    })
    // 머신러닝 로깅용
    socket.on('mlLog', function(data){
        console.log("MLLOG : " + data);
    })

    // 머신러닝 트레이닝 정보 -> client
    socket.on('mlTrain', function(data){
        io.to(socket.room).emit('mlTrain', data);
    })
    // 머신러닝 테스트 정보 -> client 
    socket.on('mlTest', function(data){
        console.log(data);
        io.to(socket.room).emit('mlTest', data);
    });
    // 머신러닝 에러 정보 -> client
    socket.on('mlErr', function(data){
        console.log('mlErr');
        console.log(data);
        io.to(socket.room).emit('mlErr', data);
    });
    // 머신러닝 prophet 학습 데이터 -> client
    socket.on('prophet', function(data){
        console.log(data);
        io.to(socket.room).emit('prophet', data);
    });
    // 머신러닝 prophet 실제값 -> client
    socket.on('prophet_real', function(data){
        console.log(data);
        io.to(socket.room).emit('prophet_real', data);
    })
});

http.listen(80, function(){
    console.log('server On');
});

// nodejs => 런타임환경( 자바스크립트를 서버에서 돌리기위한 녀석 ) -> 