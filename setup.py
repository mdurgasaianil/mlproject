## with this file, we will be able to create our entire ML application as an package. That can be used anywhere even in python pypi.
## -e . present in requirements.txt which will be used to run the setup.py while requirements.txt file was executing.

from typing import List
from setuptools import find_packages,setup

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requrements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
name='mlporject',
version='0.0.1',
author='Anil',
author_email='durgaanil7@gmail.com',
packages=find_packages(), ## this function will consider as an package with which folders were having the file name is __init__.py
install_requires=get_requirements('requirements.txt')
)