# Native imports
import json

from mmcore.addons.rhino.native import random
from mmcore.addons.rhino.native.utils import *

with open("mmconfig.json") as config_file:
    mmconfig = json.load(config_file)
