import inspect
import time

import rich


class CodeTime:
    def __init__(self, name='', print_time=False):
        self.print_time = print_time
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time() - self.start
        if self.print_time:
            rich.print(dict(zip(('min', 'sec'), divmod(self.end, 60))))

        if (exc_type, exc_val, exc_tb) == (None, None, None):
            pass
        else:

            raise exc_type(exc_val, exc_tb)
