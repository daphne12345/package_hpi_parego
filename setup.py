from setuptools import setup

from my_config_selector import MyConfigSelector
from my_ei_hpi import MyEI
from my_local_and_random_search import MyLocalAndSortedRandomSearch
from my_local_search import MyLocalSearch
from fanova import fANOVAWeighted

setup(
    name='pckg_hpi_parego',
    version='1',

    url='https://github.com/daphne12345/package_hpi_parego',
    author='Daphne',

    py_modules=['my_config_selector', 'my_ei_hpi', 'my_local_and_random_search',
                'my_local_search', 'fanova'],
)