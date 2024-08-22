from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent


setup(
    name="mamsi",
    version="0.1.1",
    packages=find_packages(),
    license="BSD 3-Clause License",
    package_data={'': ['Data/Adducts/*', 'Data/ROI/*']},
    author="Lukas Kopecky",
    author_email="l.kopecky22@imperial.ac.uk",
    description="Multi-Adduct Mass Spectrometry Integration (MAMSI) is a Python package for the integration of multimodal LC-MS and MSI data.",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],

    install_requires=[
        'mbpls==1.0.4',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'seaborn',
        'networkx',
        'pyvis',
    ],

)