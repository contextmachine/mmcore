import subprocess
import threading

try:
    # Python 2.x
    from urlparse import urlparse
    from urllib import quote
except ImportError:
    # Python 3.x
    from urllib.parse import urlparse, quote
from flask import Flask, request
from flask_redis_sentinel import SentinelExtension
from flask_bootstrap import Bootstrap
from config import configure
import redis

redis_sentinel = SentinelExtension()
sentinel = redis_sentinel.sentinel

app = Flask(__name__)

configure(app)
redis_sentinel.init_app(app)
Bootstrap(app)


class MemtierThread(threading.Thread):
    def __init__(self, master_ip, master_port, redis_password=None, argument_line="", **kwargs):
        try:
            # Python 3.x
            super().__init__(**kwargs)
        except TypeError:
            # Python 2.x
            super(MemtierThread, self).__init__(**kwargs)
        self._master_ip = master_ip
        self._master_port = master_port
        self._redis_password = redis_password
        self._argument_list = argument_line.split()
        self._output = ""
        self._return_code = None
        self._process = None

    def run(self):
        self._process = subprocess.Popen(["./memtier_benchmark", "-s", self._master_ip, "-p", self._master_port, "-a",
                                          self._redis_password] + self._argument_list,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=False)
        while True:
            curr_output = self._process.stdout.readline().decode("utf-8")
            if "[RUN" in curr_output:
                temp_output = curr_output.split("[RUN")
                curr_output = "\n[RUN".join(temp_output)
            if curr_output == '':
                self._return_code = self._process.poll()
                if self._return_code != None:
                    return
            if curr_output:
                self._output = self._output + "\n" + curr_output.strip()

    def kill(self):
        if self._process:
            self._process.kill()
            self.join()
            self._process = None

    @property
    def output(self):
        return self._output

    @property
    def return_code(self):
        return self._return_code


@app.route('/execute', methods=['POST'])
def execute():
    success = False
    req = request.get_json()
    try:
        conn = get_conn_through_sentinel()
        response = conn.execute_command(*req['command'].split())
        success = True
    except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError):
        try:
            reload_username_password_from_file_system_if_needed(app)
            conn = get_conn_through_sentinel()
            response = conn.execute_command(*req['command'].split())
            success = True
        except Exception as err:
            response = 'Exception: cannot connect. %s' % str(err)
            app.logger.exception("execute err")
    except Exception as err:
        response = 'Exception: %s' % str(err)
