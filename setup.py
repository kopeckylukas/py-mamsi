from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mamsi",
    version="1.0.2",
    packages=find_packages(),
    url="https://github.com/kopeckylukas/py-mamsi",
    license="BSD 3-Clause License",
    package_data={'': ['Data/Adducts/*', 'Data/ROI/*']},
    author="Lukas Kopecky",
    author_email="l.kopecky22@imperial.ac.uk",
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="Multi-Adduct Mass Spectrometry Integration (MAMSI) is a Python package for the integration of multimodal LC-MS and MSI data.",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Metabolomics',
        'Intended Audience :: Mass Spectrometry',
        'Intended Audience :: Bioinformatics',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
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