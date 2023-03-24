"""
Note:
Notice the 'async:true' specifier below. This ensures the associated run,
debug, and profile commands run this script on a non-ui thread so Rhino
UI does not get locked when script is running.
"""
#! async:true


from __future__ import absolute_import, annotations
import sys, os
import subprocess
sys.path.extend(["~/PycharmProjects/mmcore"])
os.environ["RPYC_CONFIGS"]="http://storage.yandexcloud.net/box.contextmachine.space/share/configs/rhpyc.yaml"
from mmcore.services.rhpyc import RhService


def main():
    
    RhService.run()


if __name__ == "__main__":

    sys.exit(main())
