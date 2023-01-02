import os, sys
import socket, json
import rhino3dm as rg
import json, ast
import bz2, base64
def decode(dct):
    if isinstance(dct, list):
        return [decode(geom) for geom in dct]
    elif isinstance(dct, dict):
        if "archive3dm" in dct.keys():
            return rg.GeometryBase.Decode(dct)
        elif "X" in dct.keys():
            return rg.Point3d(*list(dct.values()))
        else:
            return dict([(k, decode(v)) for k, v in dct.items()])
    elif isinstance(dct, (int,float,bool,bytes,str)):
        return dct
    else:
        return dct
#a=decode(json.loads(x))
def decompress(string):
    return decode(json.loads(bz2.decompress(base64.b16decode(string.encode())).decode()))
def encode(geoms):
    if isinstance(geoms, list):
        return [encode(geom) for geom in geoms]
    elif isinstance(geoms, dict):
           return dict([(k,encode(v)) for k,v in geoms.items()])
    elif hasattr(geoms, "Encode"):
        return geoms.Encode()
    elif isinstance(geoms, (int,float,bool,bytes,str)):
        return geoms
    elif hasattr(geoms, "ToNurbsCurve"):
        return geoms.ToNurbsCurve().Encode()
    else:
        raise TypeError("Can not encode this :(")
import pprint
def compress(data):
    return base64.b16encode(bz2.compress(json.dumps(encode(data)).encode(),compresslevel=9)).decode()
