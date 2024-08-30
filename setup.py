from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mamsi",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/kopeckylukas/py-mamsi",
    license="BSD 3-Clause License",
    package_data={'': ['Data/Adducts/*', 'Data/ROI/*']},
    author="Lukas Kopecky",
    author_email="l.kopecky22@imperial.ac.uk",
    description='The Multi-assay Mass Spectrometry Integration Project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="Multi-Adduct Mass Spectrometry Integration (MAMSI) is a Python package for the integration of multimodal LC-MS and MSI data.",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
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