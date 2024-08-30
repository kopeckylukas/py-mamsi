#


## MamsiStructSearch
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L29)
```python 
MamsiStructSearch(
   rt_win = 5, ppm = 15
)
```


---
A class for performing structural search on multi-modal MS data using.
The class allows to search for structural signatures in LC-MS data based on their m/z and RT.
These structural signatures include isotopologues and adduct patterns.



**Attributes**

* **assay_links** (list) : List of data frames containing links for each assay.
* **intensities** (numpy.ndarray) : Array of LC-MS intensity data.
* **rt_win** (int) : Retention time tolerance window.
* **ppm** (int) : Mass-to-charge ratio (m/z) tolerance in ppm.
* **feature_metadata** (pandas.DataFrame) : Data frame containing feature metadata extracted from column names.
* **structural_links** (pandas.DataFrame) : Data frame containing structural clusters.


**Args**

* **rt_win** (int, optional) : Retention time tolerance window. Defaults to 5.
* **ppm** (int, optional) : Mass-to-charge ratio (m/z) tolerance in ppm. Defaults to 15.



**Methods:**


### .load_msi
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L58)
```python
.load_msi(
   df
)
```

---
Imports MSI intensity data and extracts feature metadata from column names.


**Args**

* **df** (pandas.DataFrame) : Data frame with MSI intensity data.
    - rows: samples
    - columns: features (m/z peaks)
        Column names in the format:
            <m/z>
        For example:
            149.111


### .load_lcms
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L102)
```python
.load_lcms(
   df
)
```

---
Imports LC-MS intensity data and extracts feature metadata from column names.


**Args**

* **df** (pandas.DataFrame) : Data frame with LC-MS intensity data.
    - rows: samples
    - columns: features (LC-MS peaks)
        Column names in the format:
            <Assay Name>_<RT in sec>_<m/z>m/z
        For example:
            HPOS_233.25_149.111m/z


### .get_structural_clusters
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L150)
```python
.get_structural_clusters(
   adducts = 'all', annotate = True
)
```

---
Searches structural signatures in LC-MS data based on their m/z and RT. These structural signatures include 
isotopologues, adduct patterns and cross-assay links.


**Args**

* **adducts** (str, optional) : Define what type of adducts to . 
    Possible values are:
        - 'all': All adducts combinations (based on Fiehn Lab adduct calculator).
        - 'most-common': Most common adducts for ESI (based on Waters adducts documentation).
    Defaults to 'all'.
* **annotate** (bool, optional) : Annotate significant features based on National Phenome Centre RIO data.
    Only to be run if the data was analysed by the National Phenome Centre or analysis followed their
    conventions and protocols. For more information see https://doi.org/10.1021/acs.analchem.6b01481 
    or https://phenomecentre.org.
    Uses semi-targeted annotations for selected compounds.
    Defaults to True.


**Returns**

* **list** (pandas.DataFrame) : DataFrame of significant features with structural clusters.




### .get_correlation_clusters
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L705)
```python
.get_correlation_clusters(
   flat_method = 'constant', cut_threshold = 0.7, max_clusters = 5,
   cor_method = 'pearson', linkage_method = 'complete', metric = 'euclidean',
   **kwargs
)
```

---
Clusters features based on their correlations. The method uses hierarchical clustering to create clusters.
To flatten clusters, the method uses either a constant threshold or silhouette score.


**Args**

* Flattens clusters based on a constant threshold (cut_threshold).
    - 'silhouette': Flattens clusters based on most optimal silhouette score.
    Defaults to 'constant'.
* **cut_threshold** (float, optional) : Constant threshold for flattening clusters. Defaults to 0.7.
* **max_clusters** (int, optional) : Maximum number of clusters for silhouette method. Defaults to 5.
* **cor_method** (str {'pearson', 'kendall', 'spearman'}, optional) : Method for calculation correlations. Defaults to 'pearson'.
* **linkage_method** (str, optional) : The linkage criterion determines which distance to use between sets of observation.
    The algorithm will merge the pairs of cluster that minimise this criterion.
    - 'single': Single linkage minimises the maximum distance between observations of pairs of clusters.
    - 'complete': Complete linkage minimises the maximum distance between observations of pairs of clusters.
    - 'average': Average linkage minimises the average of the distances between all observations of pairs of clusters.
    - 'ward': Ward minimises the variance of the clusters being merged.
    - 'weighted': Weighted linkage minimises the sum of the product of the distances and the number of observations in pairs of clusters.
        Only available for 'constant' flatting method.
    - 'centroid': Centroid linkage minimises the distance between the centroids of clusters.
        Only available for 'constant' flatting method.
    - 'median': Median linkage minimises the distance between the medians of clusters.
        Only available for 'constant' flatting method.
    Defaults to 'complete'.
* **metric** (str, optional) : The distance metric to use. The metric to use when calculating distance between instances in a feature array.
    Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
    If linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix is needed as input for the fit method.
    Defaults to 'euclidean'.
flat_method (str {'constant', 'silhouette'}, optional):
    Method for cluster flattening:

### .get_structural_network
[source](https://github.com/kopeckylukas/py-mamsi/blob/main/mamsi/mamsi_struct_search.py/#L850)
```python
.get_structural_network(
   include_all = False, interactive = False, return_nx_object = False,
   output_file = 'interactive.html', labels = False, master_file = None
)
```

---
Generates a structural network graph based on the provided master file or the loaded structural links data.


**Args**

* **include_all** (bool, optional) : Whether to include all features in the network, even if they are not structurally linked to other features.
    Defaults to False.
* **interactive** (bool, optional) : Whether to display the network graph interactively using pyvis.network.
    If False, the network graph is displayed using NetworkX and Matplotlib.
    Defaults to False.
* **return_nx_object** (bool, optional) : Whether to return the NetworkX object representing the network graph edited in CytoScape. 
    Defaults to False.
* **output_file** (str, optional) : The name of the output file when displaying the network graph interactively using pyvis.network.
    Only applicable when interactive is True. 
    Defaults to 'interactive.html'.
* **labels** (bool, optional) : Whether to display labels for the nodes in the network graph.
    Only applicable when interactive is False. 
    Defaults to False.
* **master_file** (pd.DataFrame, optional) : The master file containing necessary columns for generating the network.
    This is intended for cases when structural links required manual curation (e.g. manually assigned isotopologue groups, adduct groups, etc.)
    If not provided, the function uses the loaded structural links data.
    Required columns: 
        - Feature: Feature ID (e.g. HPOS_233.25_149.111m/z)
        - Assay: Assay name (e.g. HPOS)
        - Isotopologue group (groups features with similar isotopologue patterns)
        - Isotopologue pattern (e.g. 0, 1, 2 ... N representing M+0, M+1, M+2 ... M+N)
        - Adduct group (groups features with similar adduct patterns)
        - Adduct (adduct label, e.g. [M+H]+, [M-H]-)
        - Structural cluster (groups features with similar isotopologue and adduct patterns)
        - Correlation cluster (flattened hierarchical cluster from get_correlation_clusters()
        - Cross-assay link (links features across different assays)
        - cpdName (compound name, optional)
    Defaults to None.


**Returns**

* **None**  : The NetworkX object representing the network graph, if return_nx_object is True.
    Edge weights represent the type of link between features:
        - Isotopologue: 1
        - Adduct: 5
        - Cross-assay link: 10
    Otherwise, None.
                        


**Raises**

* **RuntimeWarning**  : If no data is loaded and no master file is provided.
* **RuntimeWarning**  : If the provided master file is missing necessary columns.

---
Notes:
    - The function creates a network graph based on the provided master file or the loaded structural links data.
    - The network graph includes nodes representing features and edges representing different types of links.
    - The graph can be displayed interactively using pyvis.network or using NetworkX and matplotlib.
    - The graph can be saved as a NetworkX object if return_nx_object is True.
