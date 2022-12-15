import socket
import sys

import numpy as np

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)
while True:
    print(sys.stderr, '\nwaiting to receive message')
    data, address = sock.recvfrom(2048 ** 2)
    datta = np.asarray(np.safe_eval(data.decode()), dtype=int)
    print(sys.stderr, 'received %s bytes from %s' % (len(data), address))
    print(sys.stderr, datta * 9)

    if data:
        sent = sock.sendto(data, address)
        print(sys.stderr, 'sent %s bytes back to %s' % (sent, address))
