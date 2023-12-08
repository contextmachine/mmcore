import subprocess
import unittest


class DockerTest(unittest.TestCase):
    def case(self):
        proc = subprocess.Popen(
            list("docker run --rm --tty --name mmcore sthv/mmcore:amd64 python tests/dockertest.py".split(" ")))
        proc.communicate()
        saved = proc.stdout.read()
        proc.terminate()
        self.assertEqual(int(saved), 0)


if __name__ == "__main__":
    DockerTest().case()
