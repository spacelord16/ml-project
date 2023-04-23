from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

# This function will return the list of req..
def get_requirements(file_path:str)->List[str]:
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
    
setup(
    
    name = 'ml-project',
    version = '0.0.1',
    author = 'Aditya',
    author_email = 'deshpande.aditya16@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    
)
