import numpy
from setuptools import Extension, Distribution
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

compile_args = ["-O3"]
link_args = []
include_dirs = [numpy.get_include(), '/']

extensions = [
    Extension(
        "mmcore.geom.vec.spherical",
        ["mmcore/geom/vec/spherical.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    )
]

ext_modules = cythonize(extensions, include_path=[numpy.get_include()])
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

import os, shutil

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)
