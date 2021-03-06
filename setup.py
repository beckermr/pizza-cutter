import os
from setuptools import setup, find_packages

scripts = [
    'bin/des-pizza-cutter',
    'bin/des-pizza-cutter-prep-tile',
]

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pizza_cutter",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name='pizza_cutter',
    version=__version__,
    description="yummy survey slices",
    author="MRB and ESS",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
)
