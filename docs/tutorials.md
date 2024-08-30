# Tutorials and Training Materials

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/kopeckylukas/py-mamsi/blob/main/LICENCE)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://kopeckylukas.github.io/py-mamsi/) 
[![pages-build-deployment](https://github.com/kopeckylukas/py-mamsi/actions/workflows/pages/pages-build-deployment/badge.svg)](https://kopeckylukas.github.io/py-mamsi/)


## Training Materials
You can find all MAMSI training materials by visiting our [MAMSI Tutorials](https://github.com/kopeckylukas/py-mamsi-tutorials) repository. 

The `MamsiPls` class inherits from the `mbpls` package [[1](./index.md/#references)]. For more information on MB-PLS, please visit [mbpls Documentation](https://mbpls.readthedocs.io/en/latest/index.html).



## Tutorial Example - Classification
The Multi-Assay Mass Spectrometry Integration ([MAMSI](https://github.com/kopeckylukas/py-mamsi)) workflow allows for integrative analysis of multiple metabolomics LC-MS assays data. The MAMSI workflow utilises a multi-block partial-least squares (MB-PLS) discriminant analysis algorithm, which allows for the integration of multiple assays and the subsequent identification of the most significant predictors (features). The identification of statistically significant predictors is done using a multi-block version of the variable importance in projection (MB-VIP) procedure coupled with permutation testing. This enables us to obtain empirical p-values for each feature across all assays.
MAMSI also offers an easy interpretation of significant features. This is done by grouping of the significant features based on their structural properties (mass-to-charge ratio and retention time) and compared to their correlations. This can be visualised by a network plot.

This notebook showcases the use of the MAMSI workflow for the prediction (classification) of the biological sex of patients within the AddNeuroMed cohort [[3](index.md/#references)] - dataset of Alzheimer's disease patients. For this task, we will use 3 metabolomics blood serum assays. The assays were processed by the [National Phenome Centre](https://phenomecentre.org) following the NPC protocol [[6](index.md/#references)]. Subsequently, data were pre-processed using XCMS [[7](index.md/#references)] and nPYc toolbox [[8](index.md/#references)].

Assays Overview 

| Assay | Number of features | Description                                                                   |
| ----- | ------------------ | ----------------------------------------------------------------------------- |
| HPOS  | 681                | Hydrophilic interaction liquid chromatography (**HILIC**) positive ionisation |
| LPOS  | 4,886              | Lipidomic reversed  phase chromatography positive ionisation                  |
| LNEG  | 2,091              | Lipidomic reversed phase chromatography negative ionisation                   |


Outcome variable: Biological Sex

| Class  | Number of samples |
| ------ | ----------------- |
| Male   | 283               |
| Female | 294               |

### Load Packages

```python 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mamsi.mamsi_pls import MamsiPls
from mamsi.mamsi_struct_search import MamsiStructSearch
from matplotlib import pyplot as plt
```

### Load Sample Data
<br> Data used within this quickstart guide originate from the the AddNeuroMed [[1](./index.md/#references)] cohort - dataset of Alzheimer's disease patients. 
You can download the sample data from this [link](https://github.com/kopeckylukas/py-mamsi-tutorials/tree/main/sample_data).

```python
metadata = pd.read_csv('../sample_data/alz_metadata.csv')
# The PLS algorithm requires the response variable to be numeric. 
# We will encode the outcome "Gender" (Biological Sex) as 1 for female and 0 for male subjects. 
y = metadata["Gender"].apply(lambda x: 1 if x == 'Female' else 0)

# Import LC-MS data
# Add prefix to the columns names. This will be crucial for interpreting the results later on.
hpos = pd.read_csv('./sample_data/alz_hpos.csv').add_prefix('HPOS_')
lpos = pd.read_csv('./sample_data/alz_lpos.csv').add_prefix('LPOS_')
lneg = pd.read_csv('./sample_data/alz_lneg.csv').add_prefix('LNEG_')
```

### Split data
Split the dataset into training and testing subsets. The training subset will be used to fit the model, for cross-validation and for estimation of the number of latent variables. Then, the testing subset will be used as an independent dataset to assess the performance of the model. This will ensure that the model is not over-fitted to the data and that it can predict the outcome of the samples. 

```python
# Split data in the ratio 90:10 for training and testing respectively.
hpos_train, hpos_test, y_train, y_test = train_test_split(hpos, y, test_size=0.1, random_state=42)

# Split the other two blocks based on the indices of the hpos block.
lpos_train = lpos.iloc[hpos_train.index,:]
lneg_train = lneg.iloc[hpos_train.index,:]

lpos_test = lpos.iloc[hpos_test.index,:]
lneg_test = lneg.iloc[hpos_test.index,:]
```

### Fit Model
#### Fit MB-PLS Model and Estimate LVs
As an example, we will start by fitting a MB-PLS model (from the MamsiPls class) with 1 component/latent variable (LV) and using the standard scaler. As a result, we will obtain super scores, block loadings and scores and block importances. Note that MamsiPls inherits its behaviour from the [mbpls](https://pypi.org/project/mbpls/) [[1](./index.md/#references)] model which, by default, uses the *NIPALS* algorithm. To see all possible configurations for the **MamsiPls** and **mbpls** models, see the [mbpls documentation](https://mbpls.readthedocs.io/en/latest/).

We can then estimate the number of latent variables in the model. The MamsiPls class provides method ```MamsiPls.estimate_lv()``` to estimate number of LVs in the model using k-fold cross-validation (CV). The k-fold CV is repeated k-times corresponding to number of LVs in the most complex model. The lowest possible number of LVs where the model stabilised (model performance did not rise by adding more LVs) was selected as the final model.

For the classification task, you can choose from `'auc', 'precision', 'recall', 'f1', 'accuracy'` metrics to perform the LV estimation on the mean value of validation CV splits.
```python 
mamsipls = MamsiPls(n_components=1)
mamsipls.fit([hpos_train, lpos_train, lneg_train], y_train)

# Estimate the number of latent variables in you model
mamsipls.estimate_lv([hpos_train, lpos_train, lneg_train], y_train, metric='auc')
```
![LV estimation result](./images/lv_estimation.png)
<br>The LV estimation result shows that the model has 6 latent variables/components. Adding more LVs to the model could lead to overfitting. 

#### Evaluate Final Model
We can evaluate the performance of the model by predicting the outcome on an independent (testing) dataset that has not been used for model training. For this, we will use the 'testing' subset that we obtained during the train-test-split. 

We can get the performance scores by calling the ```mamsipls.evaluate_class_model()``` method.

```python
predicted = mamsipls.evaluate_class_model([hpos_test, lpos_test, lneg_test], y_test.array)
```
![Confusion Matrix](./images/confusion_matrix.png)

| Metric              | Score |
| ------------        | ----- |
| Accuracy            | 0.966 |
| Recall              | 1.0   |
| Specificity         | 0.943 |
| F<sub>1</sub> Score | 0.971 |
| AUC                 | 0.933 |

The scores and confusion matrix above indicate that the model performance has improved. If we are happy with such model, we can start with model interpretation. 


### Estimate Feature Importance
We can start with reviewing the Multi-Block Variable Importance in Projection (MB-VIP) scores [[2](./index.md/#references)].
The MB-VIP metric is the sum (weighted by the amount of variance of Y explained by each respective component) of the squared weight values. It provides a summary of the importance of a variable accounting for all weight vectors. VIPs are bounded between 0 (no effect) and infinity. Because it is calculated from the weights *w*, for PLS models with a single component, these are directly proportional to *w<sup>2</sup>*. The VIP metric has the disadvantage of pooling together *w* vectors from components which contribute a very small magnitude to the model's *R<sup>2</sup>Y*.

#### Multiblock Variable Importance in Projection

```python
mb_vip = mamsipls.mb_vip(plot=True)
```
![Multiblock VIP](./images/mb-vip.png)

Unfortunately, the assessment of variable importance in MB-PLS multivariate models is not straightforward, given the choice of parameters and their different interpretations, especially in models with more than 1 LV. To obtain a ranking of variables from the data matrix X associated with Y, we recommend using permutation testing coupled with the MB-VIP metric to estimate the empirical p-values for each variable.

#### Permutation Testing
You can perform permutation testing using the `MamsiPls.mb_vip_permtest()` method. We recommend to perform at least 10,000 permutations, but ideally >100,000 for a good p-value estimate. You can find pre-calculated p-values at [link](https://github.com/kopeckylukas/py-mamsi-tutorials/tree/main/sample_data).

We can review the empirical null distribution of the MB-VIP scores of a statistically non-significant feature (1) and compare it to a statically significant feature (5769).

In both cases, the dashed line indicates the MB-VIP score for each feature of the observed (non-permuted) model. 

The empirical p-values for the *i* <sup>th</sup> feature are calculated by dividing the number of MB-VIP scores of the *null (permuted)* models for the *i* <sup>th</sup> feature **higher** than the *observed* MB-VIP score for *i* <sup>th</sup> feature, by the total number of permutations.


```python
p_vals, null_vip = mamsipls.mb_vip_permtest([hpos, lpos, lneg], y, n_permutations=10000, return_scores=True)
```
![Null Models Distribution](./images/null_models_distribution.png)

*Note that the pre-calculated `null_vip` file contains MB-VIP scores for first 400 null models (permutations) only so the p-value displayed on the plot below does not correspond with the plot itself.*

### Interpret Statistically Significant Features
```python
# merge the LC-MS data into a single data frame
x = pd.concat([hpos, lpos, lneg], axis=1)

# Select features with p-value < 0.01.
# You can also apply multiple testing correction methods to adjust the p-value threshold.
mask = np.where(p_vals < 0.01)
selected = x.iloc[:, mask[0]]
```
You can use MAMSI Structural Search tool (`MamsiStructSearch()`) to help you understand the nature of statistically significant features. 

Firstly, all features are split into retention time (*RT*) windows of 5 seconds intervals, then each RT window is searched for isotopologue signatures by searching mass differences of 1.00335 *Da* between mass-to-charge ratios (*m/z*) of the features; if two or more features resemble a mass isotopologue signature then they are grouped together. This is followed by a search for common adduct signatures. This is achieved by calculating hypothetical neutral masses based on common adducts in electrospray ionisation. If hypothetical neutral masses match for two or more features within a pre-defined tolerance (15 *ppm*) then these features are grouped together. Overlapping adduct clusters and isotopologue clusters are then merged to form structural clusters. Further, we search cross-assay clusters using [M+H]<sup>+</sup>/[M-H]<sup>-</sup> as link references. Additionally, our structural search tool, that utilises region of interest [(ROI) files](https://github.com/phenomecentre/npc-open-lcms) from peakPantheR [[4](./index.md/#references)], allows for automated annotation of  some features based on the *RT* for a given chromatography and *m/z*.
   
#### MAMSI Structural Search Tool
```python
# First, we need to define the MamsiStructSearch object 
# and choose the tolerances for the retention time and m/z matching.
struct = MamsiStructSearch(rt_win=5, ppm=10)

# Load Selected LC-MS Features
struct.load_lcms(selected)

struct.get_structural_clusters(annotate=True)
pd.set_option("display.max_rows", 15)
```
We can then perform the structural search. 
Note, use the `annotate=True` to get the annotation for selected features 
only if your LC-MS data originate from the National Phenome Centre or follow the NPC protocol.


We also need to perform hierarchical correlation clustering between selected features. 
You can choose either the silhouette method or define a straight cut-off on the dendrogram
```python
struct.get_correlation_clusters(flat_method='silhouette', max_clusters=11)
```
Best number of clusters based on silhouette score: 8 </br>
Silhouette score for 8 clusters: 0.2436798413177305

![Heatmap](./images/correlation_heatmap.png)
![Silhouette Plot](./images/silhouette_plot.png)


Finally we can visualise the structural relationships using a network plot. 
The different node colours represent different flattened hierarchical correlation clusters, while the edges between nodes identify their structural links. You can also save the network as an NX object and review in [Cytoscape](https://cytoscape.org) to get better insight on what the structural relationship between individual features are (e.g. adduct links, isotopologues, cross-assay links).
```python
network = struct.get_structural_network(include_all=True, interactive=False, labels=True, return_nx_object=True)
```
![Network](./images/network.png)
