![MAMSI_logo](./images/MAMSI_logo.png)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/kopeckylukas/py-mamsi/blob/main/LICENCE)
[![pages-build-deployment](https://github.com/kopeckylukas/py-mamsi/actions/workflows/pages/pages-build-deployment/badge.svg)](https://kopeckylukas.github.io/py-mamsi/)
[![PyPI version](https://img.shields.io/pypi/v/mamsi.svg)](https://pypi.org/project/mamsi/)
[![DOI](https://zenodo.org/badge/823594568.svg)](https://zenodo.org/doi/10.5281/zenodo.13619607)

[MAMSI](https://github.com/kopeckylukas/py-mamsi) is a Python framework designed for the integration of multi-assay mass spectrometry datasets. 
In addition, the MAMSI framework provides a platform for linking statistically significant features of untargeted multi-assay liquid chromatography – mass spectrometry (LC-MS) metabolomics datasets into clusters defined by their structural properties based on mass-to-charge ratio (*m/z*) and retention time (*RT*).

*N.B. the framework was tested on metabolomics phenotyping data, but it should be usable with other types of LC-MS data.*

## Overview
### Features
- Data integration analysis using the Multi-Block Partial Least Squares (MB-PLS) [[1](#references)] algorithm. The `MamsiPls` class inherits from the `mbpls` package [[1](./index.md/#references)]. For more information on MB-PLS, please visit [mbpls Documentation](https://mbpls.readthedocs.io/en/latest/index.html).
- Multi-Block Variable Importance in Projection (MB-VIP) [[2](#references)].
- Estimation of statistically significant features (variables) using MB-VIP and permutation testing.
- Linking significant features into clusters defined by structural properties of metabolites.
- Feature network links.
- Annotation of untargeted LC-MS features (only supported for assays analysed by the National Phenome Centre).

### Sources and Materials
The package source code is accessible via GitHub at [https://github.com/kopeckylukas/py-mamsi](https://github.com/kopeckylukas/py-mamsi)

Training materials including sample data can be found at [https://github.com/kopeckylukas/py-mamsi-tutorials](https://github.com/kopeckylukas/py-mamsi-tutorials).

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
- joblib

## User Installation 
### Installing with Pip
You can install MAMSI from PyPI using pip: 
```bash
pip install mamsi
```

### Installing from source-code
You can install MAMSI from source code given you have installed both Python (>=3.9) and PIP.

First, clone the repository from GitHub to your computer. You can use following commands if you have a version of Git installed on your computer:

```bash
git clone https://github.com/kopeckylukas/py-mamsi

cd py-mamsi
```

When you are in the cloned project folder, type the following code to install MAMSI and all dependencies: 
```bahs
pip install .
```

**Alternatively**, you can install dependencies using pip and MAMSI using Python:
```bash
pip install -r requirements.txt
python setup.py develop
```

# Issues and Collaboration
Thank you for supporting the MAMSI project. MAMSI is an open-source software and welcomes any form of contribution and support.

## Issues
Please submit any bugs or issues via the project's GitHub [issue page](https://github.com/kopeckylukas/py-mamsi/issues) and any include details about the (```mamsi.__version__```) together with any relevant input data/metadata. 

## Collaboration
### Pull requests
You can actively collaborate on MAMSI package by submitting any changes via a pull request. All pull requests will be reviewed by the MAMSI team and merged in due course. 

### Contributions
If you would like to become a contributor on the MAMSI project, please contact [Lukas Kopecky](https://profiles.imperial.ac.uk/l.kopecky22).

# Acknowledgement
This package was developed as part of Lukas Kopecky's PhD project at [Imperial College London](https://www.imperial.ac.uk/metabolism-digestion-reproduction/research/systems-medicine/), funded by [Waters UK](https://www.waters.com/nextgen/gb/en.html). It is free to use, published under BSD 3-Clause [licence](./LICENCE).

The authors of this package would like to acknowledge the authors of the [mbpls](https://pypi.org/project/mbpls/) package [[1](#references)] which became the backbone of MAMSI. For more information on MB-PLS, please visit [MB-PLS Documentation](https://mbpls.readthedocs.io/en/latest/index.html).

Further, we would like to thank Prof Simon Lovestone and Dr Shivani Misra for allowing us to use their data, AddNeuroMed [[3](#references)] and MY Diabetes [[5](#references)] respectively, for the development of this package. 

# Citing us
If you use MAMSI in a scientific publication, we would appreciate citations. 

## Release
```
@misc{MAMSI2024,
  author       = {Lukas Kopecky, Elizabeth J Want, Timothy MD Ebbels},
  title        = {MAMSI: Multi-Assay Mass Spectrometry Integration},
  year         = 2024,
  url          = {https://doi.org/10.5281/zenodo.13619607},
  note         = {Zenodo. Version 1.0.0},
  doi          = {10.5281/zenodo.13619607}
}
```

## Publication
The MAMSI publication is currently under the review process. 

# References
[1] A. Baum and L. Vermue, "Multiblock PLS: Block dependent prediction modeling for Python," *J. Open Source Softw.*, vol. 4, no. 34, 2019, doi: [10.21105/joss.01190](https://joss.theoj.org/papers/10.21105/joss.01190).

[2] C. Wieder *et al.*, "PathIntegrate: Multivariate modelling approaches for pathway-based multi-omics data integration," *PLOS Comput. Biol.*, vol. 20, no. 3, p. e1011814, Mar 2024, doi: [10.1371/journal.pcbi.1011814](https://pubmed.ncbi.nlm.nih.gov/38527092/).

[3] S. Lovestone *et al.*, "AddNeuroMed—The European Collaboration for the Discovery of Novel Biomarkers for Alzheimer's Disease," *Ann. N. Y. Acad. Sci*, vol. 1180, no. 1, pp. 36-46, 2009, doi: [10.1111/j.1749-6632.2009.05064.x](https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/j.1749-6632.2009.05064.x).
[4] A. M. Wolfer *et al.*, "peakPantheR, an R package for large-scale targeted extraction and integration of annotated metabolic features in LC–MS profiling datasets," *Bioinformatics*, vol. 37, no. 24, pp. 4886-4888, 2021, doi: [10.1093/bioinformatics/btab433](https://academic.oup.com/bioinformatics/article/37/24/4886/6298587).

[5] S. Misra *et al.*, "Systematic screening for monogenic diabetes in people of South Asian and African Caribbean ethnicity: Preliminary results from the My Diabetes study," presented at the *Diabet. Med.*, Mar 2018.

[6]  M. Lewis *et al.*, “An Open Platform for Large Scale LC-MS-Based Metabolomics ,” *ChemRxiv*, 2022. doi: [10.26434/chemrxiv-2022-nq9k0](https://chemrxiv.org/engage/chemrxiv/article-details/61ebd6fa0716a8529e3823dc).

[7] C. A. Smith *et al.*, "XCMS:  Processing Mass Spectrometry Data for Metabolite Profiling Using Nonlinear Peak Alignment, Matching, and Identification," *Anal. Chem.*, vol. 78, no. 3, pp. 779-787, Feb 2006, doi: [10.1021/ac051437y](https://doi.org/10.1021/ac051437y).

[8] C. J. Sands *et al.*, "The nPYc-Toolbox, a Python module for the pre-processing, quality-control and analysis of metabolic profiling datasets," *Bioinformatics*, vol. 35, no. 24, pp. 5359-5360, 2019, doi: [10.1093/bioinformatics/btz566](https://doi.org/10.1093/bioinformatics/btz566).


[def]: /images/MAMSI_logo.png

# Version History

## v1.0.4
**New Features**

- Parallelised `.kfold_cv()`
- Parallelised `.montecarlo_cv()`
- Parallelised `.estimate_lv()`
- Parallelised `.mb_vip_permtest()`

## v1.0.3
**New Features** 

- *k*-fold cross-validation implemented as a method `.kfold_cv()` that can be used for model performance evaluation. This method includes GroupKFold option.
- Monte Carlo cross-validaton (MCCV), also nown as 'random sampling cross-validation' implemented as a method `.montecarlo_cv()` that can be used for model performance evaluation.
- `.estimate_lv()` method now allows to choose between *k*-fold CV and MC-CV using parameter `method`

**Bug Fixes and Behavioural Changes**
- Plot title for `.block_importance()` fixed.
- For regression analysis, MSE metric changed to RMSE
- For `.estimate_lv()` method, parameter `y_continuous=False` was replaced with `classification=True` 


## v1.0.2
**New Features**

- New method 'MamsiPls.block_importance()': Calculate the block importance for each block in the multiblock PLS model and plot the results.

**Minor Bug Fixes and Behavioural Changes**

- Behavioural changes for `MamsiPls.mb_vip()`: The MB-VIP plot is now rendered by default, scores are not returned by default. New default arguments (plot=True, get_scores=False).
- Argument changes for `MamsiPls.estimate_lv()`: Old Arguments (no_folds, n_components) changed to (n_slplits, max_components) respectively. 
- Plots: 'Verdana' is no longer the default font. The default font changed to Matplotlib default 'DejaVu Sans'.
- Updates to `MamsiStructSearch` class to comply with future warnings - Pandas 3.0.


## v1.0.1
**Minor Bugs Fixes** 

- Fixes instances where flattened correlation clusters were misaligned to structural clusters.
- Readme licence badge links directly to GitHub licence file (URL).


## v1.0.0
**Initial Release**
