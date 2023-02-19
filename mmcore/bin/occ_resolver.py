import platform
import subprocess as sb
from collections import namedtuple
from enum import Enum

OsAndArch = namedtuple("OsAndArch", ["system", "arch", "proc"])


class InstallTypes2(tuple, Enum):
    ARM_MACOS_IN_DOCKER = OsAndArch('Linux', 'aarch64', '')
    ARM_MACOS = OsAndArch("Darwin", 'arm64', 'arm')
    INTEL_MACOS = OsAndArch("Intel", 'x86_64', 'x86_64')
    X64_LINUX = OsAndArch("Linux", 'x86_64', 'x86_64')
    DOCKER_LINUX = OsAndArch("Linux", 'x86_64', '')


def resolve_pythonocc():
    pair = InstallTypes2(OsAndArch(platform.system, platform.machine, platform.processor))
    match pair:
        case InstallTypes2.ARM_MACOS_IN_DOCKER:
            print("We are in Macos Docker Linux")
            sb.Popen(["wget",
                      "http://storage.yandexcloud.net/box.contextmachine.space/share/packages/conda/pythonocc-core.conda"])
            sb.Popen(["conda", "install", "-n", "base", "--use-local", "pythonocc-core.conda"])
        case _:
            print(f"We are in {pair} (base case)")
            sb.Popen(["conda", "install", "-n", "base", "-c", "conda-forge", "pythonocc-core"])


if __name__ == "__main__":
    resolve_pythonocc()
