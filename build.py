import platform
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
define_macros = [
    ('VOID', 'void'),
    ('REAL', 'double'),
    ('NO_TIMER', 1),
    ('TRILIBRARY', 1),
    ('ANSI_DECLARATORS', 1),
]
if sys.platform == 'darwin':
    print('Darwin')
    compile_args += ['-mcpu=native']

# see pyproject.toml for other metadata




extensions = [

    Extension(
        "mmcore.geom.nurbs",
        ["mmcore/geom/nurbs.pyx"],
        language="c++",
        extra_compile_args= ["-std=c++17"]+compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs),
     Extension(
        "mmcore.numeric.binom",
        ["mmcore/numeric/binom.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs),
     #Extension(
     #   "mmcore.numeric.intersection.ssx.cydqr",
     #   ["mmcore/numeric/intersection/ssx/cydqr.pyx"],
     #
     #    language="c++",
     #   extra_compile_args=["-std=c++11"]+compile_args,
     #        extra_link_args=link_args,
     #   include_dirs=include_dirs),
    
    
    
    Extension(
        "mmcore.numeric.algorithms.cygjk",
        ["mmcore/numeric/algorithms/cygjk.pyx"],
       
         language="c++",
        extra_compile_args=["-std=c++11"]+compile_args,
             extra_link_args=link_args,
        include_dirs=include_dirs),
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

    ),Extension(
        "mmcore.numeric.plane.cplane",
        ["mmcore/numeric/plane/cplane.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),Extension(
        "mmcore.numeric.algorithms.quicksort",
        ["mmcore/numeric/algorithms/quicksort.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),Extension(
        "mmcore.numeric.intersection.ssx._ssi",
        ["mmcore/numeric/intersection/ssx/_ssi.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),Extension(
        "mmcore.geom.evaluator.surface_evaluator",
        ["mmcore/geom/evaluator/surface_evaluator.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),Extension(
        "mmcore.geom.surfaces.ellipsoid",
        ["mmcore/geom/surfaces/ellipsoid.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs

    ),   



    Extension(
        "mmcore.numeric.integrate.romberg",
        ["mmcore/numeric/integrate/romberg.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs),
    
]
#Extension(
#        "mmcore.geom.mesh.mesh",
#        ["mmcore/geom/mesh/mesh.pyx"],
#        extra_compile_args=compile_args,
#        extra_link_args=link_args,
#        include_dirs=[numpy.get_include(), 'mmcore/geom/mesh']

#    ),

compile_args = ["-O3"]
link_args = []
include_dirs = [*include_dirs,'mmcore/topo/mesh/triangle-c']
if sys.platform == 'darwin':
    link_args += ['-mno-sse', '-mno-sse2', '-mno-sse3']
extensions.append(Extension(
       "mmcore.topo.mesh.triangle.core",
       ['mmcore/topo/mesh/triangle-c/triangle.c',"mmcore/topo/mesh/triangle/core.pyx"],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=define_macros,
    include_dirs=include_dirs))

logo=rf"""
                                                
       ____ ___  ____ ___  _________  ________ 
      / __ `__ \/ __ `__ \/ ___/ __ \/ ___/ _ \
     / / / / / / / / / / / /__/ /_/ / /  /  __/
    /_/ /_/ /_/_/ /_/ /_/\___/\____/_/   \___/ 

{platform.uname()}                                       
                                                
"""




compiler_directives=dict(boundscheck=False,wraparound=False,cdivision=True,nonecheck=False,overflowcheck=False,initializedcheck=False,embedsignature=True,language_level='3str')

if __name__ == "__main__":

    print(logo)
    ext_modules = cythonize(extensions, include_path=[numpy.get_include()], compiler_directives=compiler_directives)
    dist = Distribution({"ext_modules": ext_modules})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    import os, shutil

    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
