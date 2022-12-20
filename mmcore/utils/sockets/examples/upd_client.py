import socket
import sys

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('localhost', 10000)
message = b"[[0,1,2,3,4,0],[0,1,2,3,4,0],[0,1,2,3,4,0]]"

try:

    # Send data
    print(sys.stderr, 'sending "%s"' % message, flush=True)
    sent = sock.sendto(message, server_address)

    # Receive response
    print(sys.stderr, 'waiting to receive', flush=True)
    data, server = sock.recvfrom(4096)
    print(sys.stderr, 'received "%s"' % data, flush=True)

finally:
    print(sys.stderr, 'closing socket', flush=True)
    sock.close()
