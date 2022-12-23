import socket
import sys

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('localhost', 10000)
message = b"import Rhino.Geometry as rg\n" \
          b"ln1=rg.Line(rg.Point3d(1,2,3),rg.Point3d(1,3,6)).ToNurbsCurve()\n" \
          b"ln2=rg.Line(rg.Point3d(2,2,2),rg.Point3d(2,4,6)).ToNurbsCurve()\n" \
          b"rss=rg.NurbsSurface.CreateRuledSurface(ln1,ln2)\n" \
          b"g=rss.ToJSON(None)\nprint(g)\ng"

try:

    # Send data
    # print(sys.stderr, 'sending "%s"' % message, flush=True)
    sent = sock.sendto(message, server_address)

    # Receive response
    print(sys.stderr, 'waiting to receive', flush=True)
    data, server = sock.recvfrom(4096)
    print(eval(data))

finally:
    #print(sys.stderr, 'closing socket', flush=True)
    sock.close()
