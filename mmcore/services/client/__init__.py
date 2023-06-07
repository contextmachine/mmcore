import itertools
import os
import socket

import dotenv
import requests
import rpyc
import yaml




def get_connection(url=None):
    """


    @return: connection

    [1] Basic Example:
    """
    if url is not None:
        rconf = url
    else:
        rconf = os.getenv("RPYC_CONFIGS")
    if rconf.startswith("http"):
        data = yaml.unsafe_load(requests.get(rconf).text)
    else:
        with open(rconf) as f:
            data = yaml.unsafe_load(f)
    if list(data.keys())[0] == "service":
        configs = data["service"].get("configs")
        attrs = data["service"].get("attributes")

        hosts = configs.get("hosts")
        port = attrs.get("port")
    else:

        hosts = ["localhost"]
        port = 7778

    return get_connection_by_host_port(itertools.zip_longest(hosts, [port], fillvalue=port))


def get_connection_by_host_port(*pairs):
    i = -1
    while True:
        i += 1
        host, port = pairs[i]
        try:
            #print(host)

            conn = rpyc.connect(host=host, port=port)
            conn.ping()

            if not conn.closed:
                #print(f"{host[i]}:{port} success!!!")
                rpyc_conn = conn
                break
        except ConnectionRefusedError:
            print(f"{host}:{port} fail...")
        except TimeoutError:
            print(f"{host}:{port} fail...")
        except socket.gaierror:
            print(f"{host}:{port} fail...")
        except Exception as err:
            print(f"{host}:{port} fail...\n{err}")
    return rpyc_conn
