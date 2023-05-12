import json
import typing
from dataclasses import asdict
from fastapi.middleware.cors import CORSMiddleware
import strawberry

objdict = dict()
geomdict = dict()
matdict = dict()

# Usage example:
# from mmcore.base.registry.fcpickle import FSDB
# from mmcore.base.basic import Object3D
# c= Object3D(name="A")
# FSDB['obj']= obj
# ...
# shell:
# python -m pickle .pkl/obj
# [mmcore] : Object3D(priority=1.0,
#                    children_count=0,
#                    name=A,
#                    part=NE) at cf3d55d7-677e-4f96-9e31-b628c3962520
#

import uvicorn, fastapi

_server_binds = []

app = fastapi.FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
from fastapi import WebSocket, WebSocketDisconnect, WebSocketException


def pull():
    br = objdict["_"]
    for o in objdict.values():
        if not o.name == "base_root":
            br.add(o)
    return br


class ServerBind():

    def __new__(cls, *args, **kwargs):
        if len(_server_binds) > 0:
            return _server_binds[0]
        import threading as th
        inst = object.__new__(cls)
        _server_binds.append(inst)

        @app.get("/", responses=objdict)
        def home():

            return strawberry.asdict(pull().get_child_three())

        @app.post("/")
        def mutate(data: dict = None):
            if data is not None:
                if len(data.keys()) > 0:

                    for k in data.keys():
                        objdict[k](**data[k])

            return strawberry.asdict(pull().get_child_three())

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                print(data)
                br = objdict["_"]
                for o in objdict.values():
                    if not o.name == "base_root":
                        br.add(o)

                await websocket.send_json(data=strawberry.asdict(br.get_child_three()))

        inst.thread = th.Thread(target=lambda: uvicorn.run("mmcore.base.registry:app", port=7711, log_level="critical"))
        inst.thread.start()
        return inst

    def stop(self):
        self.thread.join(60)


serve = ServerBind()
