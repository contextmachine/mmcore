
def setup_secrets(update=False):
    import json, subprocess, base64, shutil, os
    print("Starting setup secrets")
    if update or not os.path.isfile("env.json"):
        shutil.rmtree("secrets", ignore_errors=True)
        proc = subprocess.Popen(["git", "clone", "https://github.com/contextmachine/secrets.git"])
        proc.wait()
        shutil.move("secrets/env.json", "env.json")
    with open("env.json", "r") as f:
        data = json.load(f)
        data["CADEX_LICENSE"] = base64.b64decode(data["CADEX_LICENSE"]).decode()
        for k in data.keys():
            os.environ[k] = str(data[k])
        return data
