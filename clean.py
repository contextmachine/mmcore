

import os
import glob


def delete_files(pattern, path):
    for file in glob.glob(f"{path}/**/{pattern}", recursive=True):
        try:
            os.remove(file)
            print(f"{file} has been deleted.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")




def clean_exts(exts):
    for ext in exts:
        for source in ext.sources:
            c_file=source.replace('pyx','c')
            lib = source.replace('pyx', '*.**.so')
            try:
                os.remove(  c_file)
                print(f"{  c_file} has been deleted.")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
            print("/".join(lib.split('/')[:-1]))
            delete_files( "*.so","./"+"/".join(lib.split('/')[:-1]))

if __name__ == '__main__':
    from build import extensions
    clean_exts(extensions)
