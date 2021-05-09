from setuptools import setup, find_packages

scripts = [
    'bin/des-pizza-cutter',
    'bin/des-pizza-cutter-prep-tile',
]

setup(
    name='pizza_cutter',
    description="yummy survey slices",
    author="MRB and ESS",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
