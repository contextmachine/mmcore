from __future__ import absolute_import, annotations

import yaml
from rpyc.cli.rpyc_classic import ClassicServer

__all__ = ["RhService"]


class RhService(ClassicServer):
    ...


RhService.host = "0.0.0.0"
# RhService.ssl_certfile = f"{os.getenv('HOME')}/ssl/ca-certificates/certificate_full_chain.pem"
# RhService.ssl_keyfile = f"{os.getenv('HOME')}/ssl/ca-certificates/private_key.pem"
# RhService.logfile = f"{os.getenv('HOME')}/rhpyc.log"


import pprint

with open("/Users/andrewastakhov/PycharmProjects/mmcore/mmcore/services/rhpyc/service.yaml") as f:
    data = yaml.unsafe_load(f)
    if list(data.keys())[0] == "service" and data["service"]["name"] == "rhpyc":

        configs = data["service"].get("configs")
        attrs = data["service"].get("attributes")
        real_attrs = {}
        pattrs = "attributes: {}\n\t\n"
        pconfigs = "configs: {}\n\t\n"
        if attrs:

            for k, v in attrs.items():
                if hasattr(RhService, k):
                    setattr(RhService, k, v)
                    real_attrs[k] = v
                else:
                    print(f"miss {k}")
            pattrs.format(pprint.pformat(real_attrs, indent=4))
        if configs:
            pconfigs.format(pprint.pformat(configs, indent=4))

