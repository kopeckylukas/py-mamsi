from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mamsi",
    version="1.0.4",
    packages=find_packages(),
    url="https://github.com/kopeckylukas/py-mamsi",
    license="BSD 3-Clause License",
    license_files=["LICENSE.txt"],
    package_data={'': ['Data/Adducts/*', 'Data/ROI/*']},
    author="Lukas Kopecky",
    author_email="l.kopecky22@imperial.ac.uk",
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="Multi-Assay Mass Spectrometry Integration (MAMSI) for multimodal LC-MS data.",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],

    install_requires=[
        'mbpls==1.0.4',
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn<=1.5.2',
        'seaborn',
        'networkx',
        'pyvis',
        'joblib',
    ],

)