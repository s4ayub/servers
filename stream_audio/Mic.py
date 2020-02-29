from MicIO import MicIO
import socket
import threading
import numpy as np

class MicTransmitter:
    def __init__(self):
        self.micIO = MicIO()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def callback(self, data):
        self.socket.sendall(data)

    def start(self, host, port, device_index=None):
        try:
            self.socket.connect((host, port))
        except:
            return False
        self.micIO.listen(self.callback, device_index)
        return True

    def stop(self):
        self.socket.close()
        self.micIO.stop()

class MicReceiver:
    def __init__(self, callback):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.callback = callback

    def listen(self, host, port):
        self.socket.bind((host, port))
        self.socket.listen(1)

        (clientsocket, address) = self.socket.accept()
        print('Mic connection from %s,%d' % address)

        while True:
            data = clientsocket.recv(MicIO.FramesPerBuffer)
            if not data:
                print('Connection was closed.')
                break

            data = np.frombuffer(data, dtype=np.int16).tolist()
            if self.callback:
                self.callback(data)

        self.socket.close()
