import subprocess as sp
from kubernetes import client, config

config.load_config()
import uvicorn, uvloop
uvloop.new_event_loop()

class MModelConnector:
    port: int
    active_proc: sp.Popen

    def __init__(self, port: int = 8080, timeout: int = 10):

        self.port = port

        import threading as th

        self.thread = th.Thread(target=self.connect)
        self.thread.start()
        self.timeout = timeout

    @property
    def command(self):
        return f"kubectl proxy --port={self.port} &"

    async def connect(self):
        import subprocess as sp
        async with sp.Popen(self.command.split(" "), sp.PIPE) as self.active_proc:
            self.active_proc.communicate()

    def __enter__(self):
        try:
            return self
        except Exception as err:
            self.active_proc.kill()
            self.__exit__(err.__cause__, err.args, err.__traceback__)

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.active_proc.__exit__(exc_type, exc_val, exc_tb)

    def kill(self):
        self.active_proc.kill()

    def wait(self):
        self.active_proc.wait()

    def join(self):
        self.thread.join()

    @property
    def name(self):
        return self.thread.name

    @name.setter
    def name(self, v):
        self.thread.name = v


from mmcore.base.basic import Delegate


@Delegate(client.AppsV1Api)
class MModelKubeProxy:
    ...
