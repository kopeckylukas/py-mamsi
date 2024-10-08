#


## MamsiPls
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L22)
```python 
MamsiPls(
   n_components = 2, full_svd = False, method = 'NIPALS', standardize = True,
   max_tol = 1e-14, nipals_convergence_norm = 2, calc_all = True, sparse_data = False,
   copy = True
)
```


---
A class that extends the MB_PLS class by extra methods convenient in Chemometrics and Metabolomics research. 
It is based on MB-PLS package: Baum et al., (2019). Multiblock PLS: Block dependent prediction modeling for Python.
This wrapper has some extra methods convenient in Chemometrics and Metabolomics research.

For a full list of methods, please refer to the MB-PLS class [documentation](https://mbpls.readthedocs.io/en/latest/index.html).


**Args**

* **n_components** (int, optional) : A number of Latent Variables (LV). Defaults to 2.
* **full_svd** (bool, optional) : Whether to use full singular value decomposition when performing the SVD method. 
    Set to False when using very large quadratic matrices (X). Defaults to False.
* **method** (str, optional) : The method used to derive the model attributes. Options are 'UNIPALS', 'NIPALS', 'SIMPLS', 
    and 'KERNEL'. Defaults to 'NIPALS'.
* **standardize** (bool, optional) : Whether to standardise the data (Unit-variance scaling). Defaults to True.
* **max_tol** (float, optional) : Maximum tolerance allowed when using the iterative NIPALS algorithm. Defaults to 1e-14.
* **nipals_convergence_norm** (int, optional) : Order of the norm that is used to calculate the difference of 
    the super-score vectors between subsequential iterations of the NIPALS algorithm. 
    Following orders are available:

    
    ord   | norm for matrices            | norm for vectors             |
    ----- | ---------------------------- | ---------------------------- |
    None  | Frobenius norm               |  2-norm                      |
    'fro' | Frobenius norm               | --                           |
    'nuc' | nuclear norm                 | --                           |
    inf   | max(sum(abs(x), axis=1))     | max(abs(x))                  |    
    -inf  | min(sum(abs(x), axis=1))     | min(abs(x))                  |
    0     | --                           | sum(x != 0)                  |
    1     | max(sum(abs(x), axis=0))     | as below                     |
    -1    | min(sum(abs(x), axis=0))     | as below                     |
    2     | 2-norm (largest sing. value) | as below                     |       
    -2    | smallest singular value      | as below                     |
    other | --                           | sum(abs(x)**ord)**(1./ord)   |
    Defaults to 2.
* **calc_all** (bool, optional) : Whether to calculate all internal attributes for the used method. Some methods do not need
    to calculate all attributes (i.e., scores, weights) to obtain the regression coefficients used for prediction.
    Setting this parameter to False will omit these calculations for efficiency and speed. Defaults to True.
* **sparse_data** (bool, optional) : NIPALS is the only algorithm that can handle sparse data using the method of H. Martens
    and Martens (2001) (p. 381). If this parameter is set to True, the method will be forced to NIPALS and sparse data
    is allowed. Without setting this parameter to True, sparse data will not be accepted. Defaults to False.
* **copy** (bool, optional) : Whether the deflation should be done on a copy. Not using a copy might alter the input data
    and have unforeseeable consequences. Defaults to True.


**Attributes**

* **n_components** (int) : Number of Latent Variables (LV).
* **Ts_** (array) : (X-Side) super scores [n,k]
* **T_** (list) : (X-Side) block scores [i][n,k]
* **W_** (list) : (X-Side) block weights [n][pi,k]
* **A_** (array) : (X-Side) block importances/super weights [i,k]
* **A_corrected_** (array) : (X-Side) normalized block importances (see mbpls documentation)
* **P_** (list, block) : (X-Side) loadings [i][pi,k]
* **R_** (array) : (X-Side) x_rotations R=W(PTW)-1
* **explained_var_x_** (list) : (X-Side) explained variance in X per LV [k]
* **explained_var_xblocks_** (array) : (X-Side) explained variance in each block Xi [i,k]
* **beta_** (array) : (X-Side) regression vector 𝛽 [p,q]
* **U_** (array) : (Y-Side) scoresInitialize [n,k]
* **V_** (array) : (Y-Side) loadings [q,k]
* **explained_var_y_** (list) : (Y-Side) explained variance in Y [k]



**Methods:**


### .estimate_lv
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L90)
```python
.estimate_lv(
   x, y, max_components = 10, n_splits = 5, y_continuous = False, metric = 'auc',
   plateau_threshold = 0.01, increase_threshold = 0.05, get_scores = False
)
```

---
A method to estimate the number of latent variables (LVs)/components in the MB-PLS model.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : A 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **max_components** (int, optional) : Number of components/LVs. Defaults to 10.
* **n_splits** (int, optional) : Number of splits for k-fold cross-validation. Defaults to 5.
* **y_continuous** (bool, optional) : Whether the outcome is a continuous variable. Defaults to False.
* **metric** (str, optional) : Metric to use to estimate the number of LVs; available options: ['AUC', 'precision', 'recall', 'f1'] for 
    categorical outcome variables and ['q2'] for continuous outcome variable. 
    Defaults to 'AUC'.
* **plateau_threshold** (float, optional) : Maximum increase for a sequence of LVs to be considered a plateau. Must be non-negative. Defaults to 0.01.
* **increase_threshold** (float, optional) : Minimum increase to be considered a bend. Must be non-negative. Defaults to 0.05.
* **get_scores** (bool, optional) : Whether to return measured scores as a Pandas DataFrame. Defaults to False.


**Returns**

* **DataFrame**  : Measured scores as a Pandas DataFrame.


### .evaluate_class_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L297)
```python
.evaluate_class_model(
   x, y
)
```

---
Evaluate classification MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - categorical variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .evaluate_regression_model
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L338)
```python
.evaluate_regression_model(
   x, y
)
```

---
Evaluate regression MB-PLS model using a **testing** dataset.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values - continuous variable.


**Returns**

* **array**  : Predicted y variable based on training set predictors.


### .mb_vip
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L380)
```python
.mb_vip(
   plot = True, get_scores = False
)
```

---
Multi-block Variable Importance in Projection (MB-VIP) for multiblock PLS model.

Adaptation of C. Wieder et al., (2024). PathIntegrate, doi: 10.1371/journal.pcbi.1011814.


**Args**

* **plot** (bool, optional) : Whether to plot MB-VIP scores. Defaults to False.


**Returns**

* **array**  : MB-VIP scores.


### .block_importance
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L416)
```python
.block_importance(
   block_labels = None, normalised = True, plot = True, get_scores = False
)
```

---
Calculate the block importance for each block in the multiblock PLS model and plot the results.


**Args**

* **block_labels** (list, optional) : List of block names. If block names are not provided or they do not match the number 
    of blocks in the model, the plot will display labels as 'Block 1', 'Block 2', ... 'Block n'. Defaults to None.
* **normalised** (bool, optional) : Whether to use normalised block importance. For more information see model attribute 
    ['A_Corrected_'](). Defaults to True.
* **plot** (bool, optional) : Whether to render plot block importance. Defaults to True.
* **get_scores** (bool, optional) : Whether to return block importance scores. Defaults to False.


**Returns**

* **array**  : Block importance scores.


### .mb_vip_permtest
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_pls.py/#L499)
```python
.mb_vip_permtest(
   x, y, n_permutations = 1000, return_scores = False
)
```

---
Calculate empirical p-values for each feature by permuting the Y outcome variable `n_permutations` times and
refitting the model. The p-values for each feature are calculated by counting the number of trials with
MB-VIP greater than or equal to the observed test statistic, and dividing this by `n_permutations`.

N.B. This method uses OpenMP to parallelise the code that relies on multi-threading exclusively. By default,
the implementations using OpenMP will use as many threads as possible, i.e. as many threads as logical cores.
This is available by default on systems with macOS and MS Windows.
Running this method on a computer clusters / High Performance Computing (HPC) system, including Imperial HPC, requires
additional Joblib parallelisation. A parallelised permutation test function can be found at function can be found in the 
[MAMSI Tutorials repository](https://github.com/kopeckylukas/py-mamsi-tutorials). If you are an Imperial colleague, 
do not hesitate to contact me for support on how to set up the configuration PBS file for this job.


**Args**

* **x** (array or list[array]) : All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
* **y** (array) : 1-dim or 2-dim array of reference values, either continuous or categorical variable.
* **n_permutations** (int, optional) : Number of permutation tests. Defaults to 1000.
* **return_scores** (bool, optional) : Whether to return MB-VIP scores for each permuted null model. Defaults to False.


**Returns**

* **array**  : Returns an array of p-values for each feature. If `return_scores` is True, then a matrix of MB-VIP scores
for each permuted null model is returned as well.


