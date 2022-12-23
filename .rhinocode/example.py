import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

while True:

    # print(sys.stderr, '\nwaiting to receive message')
    data, address = sock.recvfrom(2048 ** 2)
    if data:
        try:
            res = eval(data.decode().encode())
            print("{} eval {} -> {}".format(address, data, res))
            sent = sock.sendto(repr(res).encode(), address)
            print(sys.stderr, 'sent %s bytes back to %s' % (sent, address))

            sys.stdout.flush()
        except SyntaxError as err:

            res = exec(data.decode())
            print("{} exec {} -> {}".format(address, data, res))
            sent = sock.sendto(repr(res).encode(), address)

            print(sys.stderr, 'sent %s bytes back to %s' % (sent, address))
            sys.stdout.flush()
        except KeyboardInterrupt as err:
            break
        except Exception as err:
            print(err)
            sent = sock.sendto(repr(err).encode(), address)
            print(sys.stderr, 'sent %s bytes back to %s' % (sent, address))
