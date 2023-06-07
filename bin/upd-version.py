
if __name__=="__main__":
    with open(".version") as f:
        _main,major, minor=f.read().split(".")
    with open(".version", "w") as f:
        f.write(".".join([_main, major, str(int(minor) + 1)]))