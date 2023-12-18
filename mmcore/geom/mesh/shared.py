from mmcore.common.models.observer import Listener


class MeshWorker(Listener):
    def __init__(self, shm_name):
        super().__init__()
        self.shm_name = shm_name

    def notify(self, observable, **kwargs): ...
