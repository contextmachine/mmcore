import platform
import subprocess as sb
import sys
from collections import namedtuple
from enum import Enum

OsAndArch = namedtuple("OsAndArch", ["system", "arch", "proc"])


class InstallTypes2(tuple, Enum):
    ARM_MACOS_IN_DOCKER = OsAndArch('Linux', 'aarch64', '')
    ARM_MACOS = OsAndArch("Darwin", 'arm64', 'arm')
    INTEL_MACOS = OsAndArch("Intel", 'x86_64', 'x86_64')
    X64_LINUX = OsAndArch("Linux", 'x86_64', 'x86_64')
    DOCKER_LINUX = OsAndArch("Linux", 'x86_64', '')


class Conda(str, Enum):
    CONDA = "conda"
    MINICONDA = "conda"
    MAMBA = "mamba"
    MICROMAMBA = "micromamba"


def resolve_pythonocc(conda=Conda.MICROMAMBA):
    pair = InstallTypes2(OsAndArch(platform.system(), platform.machine(), platform.processor()))
    match pair:
        case InstallTypes2.ARM_MACOS_IN_DOCKER:
            #print("\n[mmcore] We are in Macos Docker Linux\n---\n")
            proc=sb.Popen(["apt","install","wget"])
            proc.wait()
            sb.Popen(["wget",
                      "http://storage.yandexcloud.net/box.contextmachine.space/share/packages/conda/pythonocc-core.conda"],
                     stdout=sys.stdout, stderr=sys.stderr)
            proc = sb.Popen([f"{conda.value}", "install","-y", "-n", "base", "--use-local", "pythonocc-core.conda"],
                            stdout=sys.stdout, stderr=sys.stderr)
            proc.communicate()

            #print("\n---\n[mmcore] done")

        case _:
            #print(f"\n[mmcore] We are in {pair} (base case)\n---\n")
            proc = sb.Popen([f"{conda.value}", "install", "-y","-n", "base", "-c", "conda-forge", "pythonocc-core"],
                            stdout=sys.stdout, stderr=sys.stderr)
            proc.communicate()

            #print("\n---\n[mmcore] done")


if __name__ == "__main__":
    resolve_pythonocc()
