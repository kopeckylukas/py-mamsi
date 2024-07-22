from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent


setup(
    name="mamsi",
    version="0.0.1",
    packages=find_packages(),
    license="BSD 3-Clause License",
    package_data={'Data': ['Adducts/*', 'ROI/*']},
    author="Lukas Kopecky",
    author_email="l.kopecky22@imperial.ac.uk",
    description="Multi-Adduct Mass Spectrometry Integration (MAMSI) is a Python package for the integration of multimodal LC-MS and MSI data.",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11'
    ],
    install_requires=[
        'mbpls',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'seaborn',
        'networkx',
        'pyvis',
        'IPython',
        'scikit-learn',
        'setuptools',
    ],

)