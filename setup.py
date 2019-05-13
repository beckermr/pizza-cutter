import os
from glob import glob
from setuptools import setup, find_packages

scripts = [
    'bin/run-metadetect-on-slices',
    'bin/des-pizza-cutter',
    'bin/gen-des-pizza-cutter-info',
    'bin/run-metadetect-on-coadd-sim']

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pizza_cutter",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

ofiles = glob('data/*')
data_files = []
for f in ofiles:
    if '~' not in f:
        data_files.append(('share/pizza-cutter', [f]))


setup(
    name='pizza_cutter',
    version=__version__,
    description="yummy survey slices",
    author="MRB and ESS",
    packages=find_packages(),
    include_package_data=True,
    data_files=data_files,
    scripts=scripts,
)
