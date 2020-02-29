import Mic
import sys

SERVER_IP='54.198.123.14'
RX_HOST='0.0.0.0'
RX_PORT=50000

def received_data(data):
    print(data)

if sys.argv[1] == 'rx':
    rx = Mic.MicReceiver(received_data)
    rx.listen(RX_HOST, RX_PORT)
elif sys.argv[1] == 'tx':
    tx = Mic.MicTransmitter()
    if tx.start(SERVER_IP, RX_PORT):
        input('Press any key to stop.')
        print('Stopping connection.')
        tx.stop()
    else:
        print('Connection refused.')
