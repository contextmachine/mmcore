import ast
import base64
import bz2
import json
import socket

import Rhino.Geometry as rg


def create_Transform(flat_arr):
    tg = rg.Transform.ZeroTransformation()
    k = 0
    for i in range(4):
        for j in range(4):
            setattr(tg, "M{}{}".format(i, j), flat_arr[k])
            k += 1
    return tg


def decode(dct):
    if isinstance(dct, (list, tuple)) and (len(dct) == 16):
        return create_Transform(dct)
    elif isinstance(dct, (list, tuple)):

        return [decode(geom) for geom in dct]
    elif isinstance(dct, dict):
        if "archive3dm" in dct.keys():
            return rg.GeometryBase.FromJSON(json.dumps(dct))
        elif "X" in dct.keys():

            return rg.Point3d(*list(dct.values()))

        else:
            return dict([(k, decode(v)) for k, v in dct.items()])
    elif isinstance(dct, (int, float, bool, bytes, str)):
        return dct
    else:
        return dct


# a=decode(json.loads(x))
def decompress(string):
    return decode(json.loads(bz2.decompress(base64.b16decode(string.encode())).decode()))


def encode(geoms):
    if isinstance(geoms, list):
        return [encode(geom) for geom in geoms]
    elif isinstance(geoms, dict):
        return dict([(k, encode(v)) for k, v in geoms.items()])
    elif hasattr(geoms, "ToJSON"):
        return ast.literal_eval(geoms.ToJSON(None))
    elif hasattr(geoms, "ToFloatArray"):
        return geoms.ToFloatArray(True)
    elif isinstance(geoms, (int, float, bool, bytes, str)):
        return geoms
    elif hasattr(geoms, "ToNurbsCurve"):
        return ast.literal_eval(geoms.ToNurbsCurve().ToJSON(None))
    else:
        raise TypeError("Can not encode this :(")


import pprint


class SelfProxy(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


def compress(data):
    return base64.b16encode(bz2.compress(json.dumps(encode(data)).encode(), compresslevel=9))


def stop(self):
    # socket.socket(socket.AF_INET, socket.SOCK_DGRAM).connect((self.hostname, self.port))
    self.socket.close()


def main(sock, server_address):
    while True:
        sock.bind(server_address)
        data, address = sock.recvfrom(65507)

        try:
            msg = decompress(data.decode())
            #print(msg)
            ctx = SelfProxy(**msg["input"])
            #print(ctx)

            if msg["py"] == "stop":

                stop(sock)

                break
            else:
                try:

                    exec(msg["py"], locals())

                    out = [eval(k) for k in msg["output"]]
                    #print(pprint.pformat(out))
                    outp = compress(out)
                    #print('response {} 200'.format(address, out, outp))
                    sent = sock.sendto(outp, address)


                except Exception as err:
                    outperr = compress({"statuscode": 400, "msg": {"err": repr(err)}})
                    sent = sock.sendto(outperr, address)
                    #print('response {} 400\n\n\t{}'.format(address, repr(err)))
        except KeyboardInterrupt as err:
            stop(sock)
            break
        except Exception as err:
            #print("Runtime Exception 500\n\n {}".format(err))
            outperr = compress({"statuscode": 500, "msg": {"critical_err": repr(err)}})
            sent = sock.sendto(outperr, address)
            stop(sock)

            break


import threading

if __name__ == "__main__":
    # Create a TCP/IP socket
    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind the socket to the port
    _server_address = ('localhost', 10082)

    #print('starting up on %s port %s' % _server_address)

    serv = threading.Thread(target=main, args=(_sock, _server_address))
    serv.start()

    #print("start sucsess")
