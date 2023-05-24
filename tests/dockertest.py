import os,sys;
if __name__=="__main__":
    try:
        import mmcore
        #print(mmcore.__version__())
        #print(0)
        sys.exit(0)
    except ImportError as err:
        #print(404)
        sys.exit(404)
    except Exception as err:
        #print(1)
        sys.exit(1)
