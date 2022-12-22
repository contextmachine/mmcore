from setuptools import setup

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
