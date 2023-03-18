import socket

import os
import sys

import requests
import rpyc
import yaml

import dotenv

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(".env"))


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
    i = -1
    while True:
        i += 1
        try:
            print(hosts[i])

            conn = rpyc.connect(host=hosts[i], port=port)
            conn.ping()

            if not conn.closed:
                print(f"{hosts[i]} success!!!")
                rpyc_conn = conn
                break
        except ConnectionRefusedError:
            print(f"{hosts[i]} fail...")
        except socket.gaierror:
            print(f"{hosts[i]} fail...")
    return rpyc_conn

