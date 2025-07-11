from setuptools import setup

setup(
    name='pckg_hpi_parego',
    packages=['hpi_parego'],
    url='https://github.com/daphne12345/package_hpi_parego',
    author='Daphne',
    install_requires=['deepcave~=1.2.1', 'numpy==1.26.4', 'scipy~=1.14.0',
        'smac~=2.3.1', 'pandas~=2.2.2', 'configspace~=1.2.1', 'shapiq~=1.3.0'],  
)