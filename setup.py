from setuptools import setup

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
    version='0.1.1',
    packages=['mmcore', 'mmcore.geom', 'mmcore.geom.tools', 'mmcore.geom.xforms', 'mmcore.geom.parametric',
              'mmcore.utils', 'mmcore.utils.sockets', 'mmcore.utils.versioning', 'mmcore.utils.pydantic_mm',
              'mmcore.utils.redis_tools', 'mmcore.addons', 'mmcore.addons.gmdl', 'mmcore.addons.opencascade',
              'mmcore.baseitems', 'mmcore.baseitems.descriptors', 'mmcore.collection', 'mmcore.collection.generics',
              'mmcore.exceptions'],
    url='',
    license='',
    author='sth-v',
    author_email='aa@contextmachine.ru',
    description=''
)
