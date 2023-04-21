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
🔫📠️💻🎚📹📟🚔🦾🦺📠️💻🎛️📱🚧🚔🦾🔫📠️🛠️🎛️📹📟🚔🦾🔫📠️👷‍🎛️🔩📟🚜⚙️🔫📠️💻🎚📹📟🚔🦾🦺📠️💻🎛️📱🚧🚔🦾🔫📠️🛠️🎛️📹📟🚔🦾🔫📠️👷‍🎛️🔩📟🚜⚙️🔫📠️💻🎚📹📟🚔🦾                                                          
------------------------------------------------------------------------------------------------------------------------

"""
      )

setup(
    name='mmcore',
    version='0.1.5',
    packages=['mmcore', 'mmcore.geom',
              'mmcore.geom.utils',
              'mmcore.geom.xforms',
              'mmcore.geom.parametric',
              'mmcore.geom.materials',
              'mmcore.geom.kernel',
              'mmcore.utils',
              'mmcore.utils.sockets',
              'mmcore.utils.versioning',
              'mmcore.utils.pydantic_mm',
              'mmcore.utils.redis_tools',
              'mmcore.addons',
              'mmcore.addons.gmdl',
              'mmcore.addons.rhino',
              'mmcore.addons.rhino.compute',
              'mmcore.addons.rhino.native',
              'mmcore.addons.mmocc',
              'mmcore.addons.mmocc.OCCUtils',
              'mmcore.services',
              'mmcore.baseitems',
              'mmcore.baseitems.descriptors',
              'mmcore.collections',
              'mmcore.collections.traversal',
              'mmcore.collections.generics',
              'mmcore.collections.multi_description',
              'mmcore.exceptions',
              'mmcore.gql',
              'mmcore.gql.client',
              'mmcore.gql.pg',
              'mmcore.gql.templates',
              'mmcore.viewer',
              'mmcore.viewer.gui',
              'mmcore.viewer.gui.chart'
              ],
    url='',
    license='',
    author='sth-v',
    author_email='aa@contextmachine.ru',
    description=''
)
