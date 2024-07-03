from setuptools import setup

setup(
    name='pckg_hpi_parego',
    packages=['hpi_parego'],
    url='https://github.com/daphne12345/package_hpi_parego',
    author='Daphne',
    install_requires=['deepcave~=1.2.1', 'numpy==1.26.4', 'scipy~=1.14.0',
        'smac~=2.1.0', 'pandas~=2.2.2', 'configspace~=0.7.1'],

    py_modules=['hpi_parego.my_config_selector', 'hpi_parego.my_ei_hpi',
                'hpi_parego.my_local_and_random_search',
                'hpi_parego.my_local_search', 'hpi_parego.fanova'],
)