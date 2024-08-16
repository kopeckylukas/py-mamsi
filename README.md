![MAMSI_logo](MAMSI_logo.png)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

MAMSI is a Python framework designed for integration of multi-assay mass spectrometry datasets. 
In addition, the MAMSI framework provides a platform for linking statistically significant features of untargeted multi-assay liquid chromatography – mass spectrometry (LC-MS) metabolomics datasets into clusters defined by their structural properties based on mass-to-charge ratio (m/z) and retention time (RT).

*N.B. the framework was tested on metabolomics phenotyping data, but it be usable with other types of LC-MS data.*

# Features
- Data integration analysis using the Multi-block Partial Least Squares (MB-PLS) algorithm.
- Multi-block variable importance in projection (MB-VIP).
- Estimation of statistically significant features (variables) using MB-VIP and permutation testing.
- Linking significant features into clusters defined by structural properties of metabolites.
- Feature network links.
- Annotation of untargeted LC-MS features (Only supported for assays analysed by the National Phenome Centre).

# Installation 
## Dependencies
- mbpls==1.0.4
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- seaborn
- networkx
- pyvis

## User Installation 
You can install MAMSI from source code code given you have installed both Python (>=3.9) and PIP.

First, clone the repository from GitHub to your computer. You can use following commands if you have a version of Git installed to your computer:

```bash
git https://github.com/kopeckylukas/py-mamsi

cd py-mamsi
```

When you are in project folder, type following code to install MAMSI and all dependencies: 
```bahs
pip install .
```

**Alternatively**, you can install dependencies using pip and MAMSI suing Python:
```bash
pip install -r requirements.txt
python setup.py develop
```

# Issues and Collaboration
Thank you for supporting the MAMSI project. MAMSI is an open-source software and welcome any forms of contribution and support.

## Issues
Please submit any bugs or issues via the project's GitHub [issue page](https://github.com/kopeckylukas/py-mamsi/issues) and any include details about the (```mamsi.__version__```) together with any relevant input data/metadata. 

## Collaboration
### Pull requests
You can actively collaborate on MAMSI package by submitting any changes via a pull request. All pull requests will be reviewed by the MAMSI team and merged in due course. 

### Contributions
If you would like to become a contributor on the MAMSI project please contact [Lukas Kopecky](https://profiles.imperial.ac.uk/l.kopecky22).

# Acknowledgement
This package was developed as part of Lukas Kopecky's PhD project at [Imperial College London](https://www.imperial.ac.uk/metabolism-digestion-reproduction/research/systems-medicine/), funded by [Waters UK](https://www.waters.com/nextgen/gb/en.html). It is free to use published under an open-source licence:

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

The authors of this package would like to acknowledge the authors of the [mbpls](https://pypi.org/project/mbpls/) package which became the backbone of MAMSI. Further, we would like thank to Prof Simon Lovestone and Dr Shivani Misra for allowing us to use their data for development of this package. 

# Citation
If you use MAMSI in a scientific publication, we would appreciate citations. 
