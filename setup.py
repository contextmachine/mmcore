import sys

from setuptools import setup

WORK_DIRECTORY = "/".join(__file__.split("/")[:-1])
sys.path.append(f"{WORK_DIRECTORY}/mmcore/bin")

print("""
Copyright (c) 2022. CONTEXTMACHINE. AEC. 
Computational Geometry, Digital Engineering and Optimizing your construction process.    
------------------------------------------------------------------------------------------------------------------------    

       ____  ____ ___  _/ /__ ____ __  __ / /__ ______    ___ __ ____ / /_  __ ___    ____     
     / ,___/ __  / __ \  ___ /  ___\ \/  /  __ / __  __ \/ __  / ,___/ __ \/ / __ \/  ___/       
    / /___/ /_/ / / / / /___/  ___ /    / /___/ / / / / / /_/ / /___/ / / / / / / /  ___/     
    \____/\___ /_/ /_/\____/\____ /_ /\_\____/_/ /_/ /_/\__,_/\____/_/ /_/_/_/ /_/\____/       


Andrew Astkhov (sth-v) aa@contextmachine.ru                                                         
------------------------------------------------------------------------------------------------------------------------

"""
      )

setup(
    name='mmcore',
    version='0.2.7',
    packages=['mmcore', 'mmcore.mmbuild', 'mmcore.geom',
              'mmcore.geom.utils',
              'mmcore.geom.transform',
              'mmcore.geom.parametric',

              'mmcore.geom.materials',
              'mmcore.geom.kernel',
              'mmcore.geom.vectors',
              "mmcore.base.geom.builder",
              'mmcore.addons',
              'mmcore.addons.gmdl',
              'mmcore.addons.rhino',
              'mmcore.addons.rhino.compute',
              'mmcore.addons.rhino.native',
              'mmcore.services',
              'mmcore.services.redis',
              'mmcore.services.redis.connect',
              'mmcore.services.redis.redis_tools',
              'mmcore.services.redis.stream',
              'mmcore.collections',
              'mmcore.collections.traversal',
              'mmcore.collections.multi_description',
              'mmcore.exceptions',
              'mmcore.node',
              'mmcore.gql',
              'mmcore.utils',
              'mmcore.gql.client',
              'mmcore.gql.pg',
              'mmcore.gql.templates',
              'mmcore.base',
              'mmcore.base.geom',
              'mmcore.base.models',
              'mmcore.base.models.gql',
              'mmcore.base.registry',
              'mmcore.base.descriptors',
              'mmcore.base.delegate'

              ],
    url='',
    license='',
    author='sth-v',
    author_email='aa@contextmachine.ru',
    description=''
)
