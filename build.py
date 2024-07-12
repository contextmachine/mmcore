import sys

if sys.platform.startswith('win'):
    import pyMSVC

    environment = pyMSVC.setup_environment()
    print(environment)

import setuptools
import numpy
# rest of setup code here
from setuptools import Extension, Distribution
from setuptools.command.build_ext import build_ext

from setuptools.command.build_py import build_py
from Cython.Build import cythonize

compile_args = ["-O3"]
link_args = []
include_dirs = [numpy.get_include()]

extensions = [
    Extension(
        "mmcore.numeric.vectors",
        ["mmcore/numeric/vectors.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ), Extension(
        "mmcore.geom.curves.deboor",
        ["mmcore/geom/curves/deboor.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ), Extension(
        "mmcore.geom.primitives",
        ["mmcore/geom/primitives.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ), Extension(
        "mmcore.numeric.calgorithms",
        ["mmcore/numeric/calgorithms.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ), Extension(
        "mmcore.geom.parametric",
        ["mmcore/geom/parametric.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs),
    Extension(
        "mmcore.geom.curves._nurbs",
        ["mmcore/geom/curves/_nurbs.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),
Extension(
        "mmcore.numeric.routines._routines",
        ["mmcore/numeric/routines/_routines.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),
Extension(
        "mmcore.geom.curves._cubic",
        ["mmcore/geom/curves/_cubic.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs


    ),
Extension(
        "mmcore.geom.implicit.tree.cbuild_tree3d",
        ["mmcore/geom/implicit/tree/cbuild_tree3d.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    )
]
#Extension(
#        "mmcore.geom.mesh.mesh",
#        ["mmcore/geom/mesh/mesh.pyx"],
#        extra_compile_args=compile_args,
#        extra_link_args=link_args,
#        include_dirs=[numpy.get_include(), 'mmcore/geom/mesh']

#    ),
if __name__ == "__main__":
    ext_modules = cythonize(extensions, include_path=[numpy.get_include(), 'mmcore/cmmcore'])
    dist = Distribution({"ext_modules": ext_modules})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    import os, shutil

    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
