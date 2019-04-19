#pip install socketIO-client-nexus==0.7.6

import sys
from socketIO_client_nexus import SocketIO
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

#fbpropeht
# import pystan
# from fbprophet import *

# History -> socket 전송용 Custom Keras History 클래스
class CustomHistory(Callback):
    def init(self, socket):
        self.train_loss = []
        self.val_loss = []
        self.socket = socket

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        data = {'column':['loss', 'val_loss'], 'data':[logs.get('loss'), logs.get('val_loss')]}

        self.socket.socket.emit('mlTrain', data)

# 소켓 통신용 케라스 클래스
class SocketKeras():
    code = None
    x_set = None
    y_set = None
    split = None
    layer = None
    meta = None
    x_scaler = None
    y_scaler = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self, socket, code, x_set, y_set, split, meta, layer):
        self.socket = socket
        self.code = code
        self.x_set = x_set
        self.y_set = y_set
        self.split = split
        self.layer = layer
        self.meta  = meta

        print("init KERAS")

        self.preprocess()
        self.makeModel()

    def preprocess(self):
        print('preporsess')

        if(self.meta['scale'] != ''):
            self.x_scaler = MinMaxScaler(feature_range=(0,1))
            self.y_scaler = MinMaxScaler(feature_range=(0,1))

            self.x_set = self.x_scaler.fit_transform(self.x_set)
            self.y_set = self.y_scaler.fit_transform(self.y_set)
        
        train_idx = int(len(self.x_set) * ((int(self.split['train']) + int(self.split['valid'])) * 0.01))
        
        self.x_train = self.x_set[:train_idx]
        self.y_train = self.y_set[:train_idx]
        
        self.x_test = self.x_set[train_idx:]
        self.y_test = self.y_set[train_idx:]

    def numericModel(self, model):
        # 입력층
        if (self.layer['input']['act'] != ''):
            model.add(Dense(int(self.layer['input']['out']), input_dim=self.x_set.shape[1], activation=self.layer['input']['act']))
        else:
            model.add(Dense(int(self.layer['input']['out']), input_dim=self.x_set.shape[1]))

        # 은닉층
        if (len(self.layer['hidden']['act']) > 0):
            for i in range(0, len(self.layer['hidden']['act'])):
                model.add(Dense(int(self.layer['hidden']['out'][i]), activation=self.layer['hidden']['act'][i]))
        # 출력층
        model.add(Dense(1))

    def modelTest(self, model):
        self.socket.changeStatus('modelTest')

        predict = model.predict(self.x_test)
        if (self.y_scaler is not None):
            predict = self.y_scaler.inverse_transform(predict)
            real = self.y_scaler.inverse_transform(self.y_test)
        else:
            real = self.y_test

        predict_list = predict.tolist()
        real_list = real.tolist()
        dist = 0

        if(len(real) >= 1000):
            dist = 10 * int(len(real)/1000)
        if(len(real) >= 10000):
            dist = 20 * int(len(real)/10000)

        for i in range(0, len(real_list)):
            if(dist > 0):
                if(i % dist != 0):
                    continue
            self.socket.socket.emit('mlTest', {'column':['real','predict'], 'data':[real_list[i][0], predict_list[i][0]]})

        self.modelErr(real, predict)

    def modelErr(self, real, predict):
        self.socket.changeStatus('modelErr')
        err_val = np.abs(predict - real)
        err_per = (np.abs(predict - real) / real) * 100
        
        err_per = err_per[np.where(np.logical_not(np.isinf(err_per)))]
        err_per = err_per[np.where(np.logical_not(np.isnan(err_per)))]

        print(err_val)
        print(err_per)
        print(predict)
        print(real)

        print("avg.err_val: {}".format(np.average(err_val)))
        print("max.err_val: {}".format(np.max(err_val)))
        print("min.err_val: {}".format(np.min(err_val)))
        print("std.err_val: {}".format(np.std(err_val)))

        print("avg.err_per: {}".format(np.average(err_per)))
        print("max.err_per: {}".format(np.max(err_per)))
        print("min.err_per: {}".format(np.min(err_per)))
        print("std.err_per: {}".format(np.std(err_per)))

        data = {'avg_err_val': round(np.average(err_val), 2),
                'max_err_val': round(np.max(err_val), 2),
                'min_err_val': round(np.min(err_val), 2),
                'std_err_val': round(np.std(err_val), 2)}

                
                # 'avg_err_per': round(np.average(err_per)),
                # 'max_err_per': round(np.max(err_per)),
                # 'min_err_per': round(np.min(err_per)),
                # 'std_err_per': round(np.std(err_per))

        self.socket.socket.emit('mlErr', data)

        self.socket.changeStatus("modelEnd")

    def makeModel(self):
        self.socket.changeStatus('modelTrain')

        model = Sequential()

        if(self.code['purCode']['code'] == 'NUMERICAL'):
            print('is Numerical')
            self.numericModel(model)
        else:
            self.numericModel(model)

        model.compile(loss=self.meta['loss'].lower(), optimizer=self.meta['opti'].lower())
        customHist = CustomHistory()
        customHist.init(self.socket)

        if(self.meta['shuffle'] == 'SHUFFLE'):
            model.fit(self.x_train, self.y_train, verbose=0, epochs=int(self.meta['epoch']), batch_size=int(self.meta['batch']), validation_split=(int(self.split['valid']) * 0.01), shuffle=True , callbacks=[customHist])
        else:
            model.fit(self.x_train, self.y_train, verbose=0, epochs=int(self.meta['epoch']), batch_size=int(self.meta['batch']), validation_split=(int(self.split['valid']) * 0.01), shuffle=False, callbacks=[customHist])

        self.modelTest(model)

# 머신러닝 소켓 통신용 클래스
class SocketML():
    #host
    #port
    #room
    #status
    #socket
    #dataSet
    #header
    #headerRow
    #df
    #x_set
    #y_set
    #meta
    #split
    #layer
    #code
    
    # 생성자(socket HOST, socket Port, socket Room ID)
    def __init__(self, host, port, room):
        self.host = host
        self.port = port
        self.room = room
        self.socketConn(host, port)
        self.status = ''

    # socket 연결 객체 생성
    def socketConn(self, host, port):
        self.socket = SocketIO(host, port)
    
    # socket 상태 변경 emit 용 메소드
    def changeStatus(self, status):
        if(self.status != status):
            self.status = status
            self.socket.emit('status', self.status)

    # socket Room에 연결 통신을 보내는 메소드
    def connRoom(self, data):
        self.socket.emit('connection',{'type':'mlModule', 'room':self.room})
        self.changeStatus('run')

    # 로깅용 메소드
    def debugLog(self, data):
        self.socket.emit('mlLog', data)

    # Room Connection이 이루어졌을때 실행하는 메소드
    def connect(self):
        print('connect')
        self.main()

    # client로 부터 데이터를 받았을 시 실행되는 메소드
    def getData(self, data):
        # 데이터셋을 받는 경우
        if(data['type'] == 'dataSet'):
            self.changeStatus('makeDataSet')
            self.dataSet = data['data']

        # 데이터셋의 헤더 데이터를 받는 경우 (데이터셋을 먼저 전달받아야함.)
        if(data['type'] == 'header'):
            self.header = data['data']['data']
            self.headerRow = data['data']['row']
            header = self.dataSet.pop(int(self.headerRow) - 1)
            self.df = pd.DataFrame(self.dataSet, columns=header)

            x_column = []
            y_column = []

            for i in self.header:
                if(i['gubun'] == 'input'):
                    x_column.append(i['name'])
                elif(i['gubun'] == 'label'):
                    y_column.append(i['name'])
            self.x_column = x_column
            self.y_column = y_column

            self.x_set = self.df[x_column].values
            self.y_set = self.df[y_column].values
            self.df = self.df[x_column + y_column]

        # 메타데이터 지정
        if(data['type'] == 'meta'):
            self.meta = data['data']

        # 데이터셋 설정 정보 지정
        if(data['type'] == 'split'):
            self.split = data['data']
        
        # 머신러닝 layer 정보 지정
        if(data['type'] == 'layer'):
            self.layer = data['data']
        
        # 머신러닝 관련 코드 정보 지정
        #
        if(data['type'] == 'code'):
            self.code = data['data']

            if(self.code['techCode']['code'] == 'MOVAVG'):
                self.makeProphet()
            else:
                self.makeML()
    # 머신러닝 시작
    def makeML(self):
        self.changeStatus('makeML')
        keras = SocketKeras(socket = self
                            , code = self.code
                            , x_set = self.x_set
                            , y_set = self.y_set
                            , layer = self.layer
                            , meta = self.meta
                            , split = self.split)

    def makeProphet(self):
        # m = Prophet()
        self.df.columns = ['ds', 'y']
        # self.df['ds'] = pd.to_datetime(self.df.ds, format='%Y-%m-%d')

        # m.fit(self.df)
        # future = m.make_future_dataframe(periods=90)
        # forecast = m.predict(future)
        forecast = pd.read_csv('./forecast.csv', header=0)
        self.changeStatus('makeProphet')
        self.socket.emit('prophet', forecast.to_csv())
        self.socket.emit('prophet_real', self.df.to_csv())

    def main(self):
        # 최초 연결 ( 방 접속 )
        self.socket.on('connection', self.connRoom)
        # 데이터 전달받기
        self.socket.on('sendData', self.getData)
        self.socket.wait(seconds=30)

socket = SocketML('http://localhost', 80, sys.argv[1])
socket.connect()

sys.exit()
