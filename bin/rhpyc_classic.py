"""
Note:
Notice the 'async:true' specifier below. This ensures the associated run,
debug, and profile commands run this script on a non-ui thread so Rhino
UI does not get locked when script is running.
"""
#! async:true

from __future__ import absolute_import, annotations

FREECADPATH = '/Applications/FreeCAD.app/Contents/Resources/lib'
import yaml, sys
try:
    from rpyc.cli.rpyc_classic import ClassicServer
except ImportError:
    import subprocess as sp
    proc=sp.Popen(["pip","install","rpyc"])
    proc.communicate()

sys.path.append(FREECADPATH)
__all__ = ["MmService", "RhService"]


class MmService(ClassicServer):
    def __init_subclass__(cls, configs=None, **kwargs):
        if configs is not None:
            os.environ["RPYC_CONFIGS"] = configs
        if os.getenv("RPYC_CONFIGS").startswith("http:") or os.getenv("RPYC_CONFIGS").startswith("https:"):
            import requests
            data = yaml.unsafe_load(requests.get(os.getenv("RPYC_CONFIGS")).text)
        else:
            with open(os.getenv("RPYC_CONFIGS")) as f:
                data = yaml.unsafe_load(f)

        if list(data.keys())[0] == "service" and data["service"]["name"] == os.getenv("RPYC_CONFIGS").split("/")[-1]:

            configs = data["service"].get("configs")
            attrs = data["service"].get("attributes")
            real_attrs = {}

            pattrs = "attributes: {}\n\t\n"
            pconfigs = "configs: {}\n\t\n"
            cls.host = attrs.get("host") if attrs.get("host") is not None else '0.0.0.0'
            cls.port = attrs.get("port") if attrs.get("port") is not None else 7777

            if attrs:

                for k, v in attrs.items():
                    if hasattr(cls, k):
                        setattr(cls, k, v)
                        real_attrs[k] = v
                    else:
                        #print(f"miss {k}")
                pattrs.format(pprint.pformat(real_attrs, indent=4))
            if configs:
                pconfigs.format(pprint.pformat(configs, indent=4))
            cls.__init_subclass__(**kwargs)


# RhService.ssl_certfile = f"{os.getenv('HOME')}/ssl/ca-certificates/certificate_full_chain.pem"
# RhService.ssl_keyfile = f"{os.getenv('HOME')}/ssl/ca-certificates/private_key.pem"
# RhService.logfile = f"{os.getenv('HOME')}/rhpyc.log"


import os, os.path

if os.path.exists("bin/socket_test.s"):
    os.remove("/tmp/socket_test.s")
os.environ["RPYC_CONFIGS"] = "http://storage.yandexcloud.net/box.contextmachine.space/share/configs/rhpyc.yaml"


class RhService(MmService, configs="http://storage.yandexcloud.net/box.contextmachine.space/share/configs/rhpyc.yaml"):
    ...


import threading, pprint


def main():
    RhService.host = "0.0.0.0"
    RhService.port = 7778
    pprint.pprint(RhService.__dict__)
    threa = threading.Thread(target=lambda: RhService.run(), name="rhpyc")
    threa.start()
  


if __name__ == "__main__":
    main()
    #print("end")

