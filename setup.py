from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent


setup(
    name="mamsi",
    version="1.0",
    packages=find_packages(),
    package_dir={'':'mamsi'},
    package_data={'Data': ['Adducts/*', 'ROI/*']},
)