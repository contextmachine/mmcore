"""
Note:
This project is work-in-progress and still in its infancy

- Reference to RhinoCommmon.dll is added by default

- You can specify your script requirements like:

    # r: <package-specifier> [, <package-specifier>]
    # requirements: <package-specifier> [, <package-specifier>]

    For example this line will ask the runtime to install
    the listed packages before running the script:

    # requirements: pytoml, keras

    You can install specific versions of a package
    using pip-like package specifiers:

    # r: pytoml==0.10.2, keras>=2.6.0
"""
import json
# ! async:true
import os

from Rhino import RhinoDoc

state = {}


def m():
    with open(f"{os.getenv('HOME')}/dev/tst.txt", "a") as log:
        while True:
            try:
                with open(f"{os.getenv('HOME')}/dev/tststate.json", "w") as log:
                    json.dump(state, log)
                    log.writelines([f"sucsess dump\n"])
                for obj in RhinoDoc.ActiveDoc.Objects:

                    if not (str(obj.Id) in state):
                        state[str(obj.Id)] = str(obj.Name)
                        log.writelines([f"Write a new object:\n", f"\t{str(obj.Name)} {str(obj.Id)}\n"])
                    else:
                        if state[str(obj.Id)] != str(obj.Name):
                            log.writelines(
                                [f"Change name event:\n", f"{state[str(obj.Id)]} -> {str(obj.Name)} {str(obj.Id)}\n"])
                            state[str(obj.Id)] = str(obj.Name)
            except KeyboardInterrupt:
                ##print("break")
                log.writelines([f"break\n"])
                break

            except Exception as err:
                ##print("break", err)
                log.writelines([f"end with err {err}\n"])
                break
        with open(f"{os.getenv('HOME')}/dev/tststate.json", "w") as log:
            json.dump(state, log)
            log.writelines([f"sucsess dump\n"])


import threading as th

thread = th.Thread(target=m)
if __name__=="__":
    thread.start()
