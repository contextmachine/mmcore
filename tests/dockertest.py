import os,sys;
if __name__=="__main__":
    try:
        import mmcore
        print(mmcore.__version__())
        sys.exit(0)
    except ImportError as err:
        sys.exit(404)
    except Exception as err:
        sys.exit(1)
