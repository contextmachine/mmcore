import json
import subprocess

from mm.conversions.cadex import cdxfb
from mm.conversions.rhino import model_from_multijson_file

CADEX_LICENSE="clicense.py"

cadex.LicenseManager.Activate(aKey)
source="/home/sthv/app/dumps/allark.3dm"
jsonpath="/home/sthv/app/rhjs/"
cadex_viewer_path="/home/sthv/cadexchanger-web-toolkit-examples/public/assets/models"
subprocess.Popen(["python",CADEX_LICENSE])
def test(name, cadex_viewer_path=cadex_viewer_path):
    model_from_multijson_file(jsonpath,modelpath=source)
    rhreader=cdxfb.readrhino(source)
    f=open(cadex_viewer_path+"/scenegraph.cdxfb","w")
    f.close()
    cdxfb.to_cadex_viewer(rhreader, target_path=cadex_viewer_path+f"/native/{name}.cdxfb/scenegraph.cdxfb")
    with open(cadex_viewer_path+"/model.json", "r")as fl:
        lst=json.load(fl)

        for i,l in enumerate(lst):
            if l["name"]==name:
                del lst[i]

                break
            else:
                continue
        lst.append({
            "name":name.replace("_"," "), "path":f"native/{name}.cdxfb"
        })


test(name="Arc_Panels")