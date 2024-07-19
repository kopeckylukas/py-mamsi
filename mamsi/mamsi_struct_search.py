# -*- coding: utf-8 -*-
#
# Author:   Lukas Kopecky <l.kopecky22@imperial.ac.uk>
#           Timothy MD Ebbels
#           Elizabeth J Want
#
# License: 3-clause BSD

import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
import pkg_resources
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns
import networkx as nx
from pyvis.network import Network
from IPython.core.display import display
from IPython.display import IFrame
from typing import Literal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


plt.rc('font', family='Verdana')


class MamsiStructSearch:
    """
    A class for performing structural search on LC-MS data using.

    Attributes:
        assay_links (list): List of data frames containing links for each assay.
        intensities (numpy.ndarray): Array of LC-MS intensity data.
        rt_win (int): Retention time tolerance window.
        ppm (int): Mass-to-charge ratio (m/z) tolerance in ppm.
        feature_metadata (pandas.DataFrame): Data frame containing feature metadata extracted from column names.

    Methods:
        load_lcms(df): Imports LC-MS intensity data and extracts feature metadata from column names.
        get_structural_clusters(adducts='all', annotate=True): Searches structural
            signatures in LC-MS data based on their m/z and RT.
        get_correlation_clusters(visualise=True): Find correlation clusters for MB-PLS features.
    """

    def __init__(self, rt_win=5, ppm=15):
        """
        Initialise MAMSI structural search class.

        Args:
            rt_win (int, optional): Retention time tolerance window. Defaults to 5.
            ppm (int, optional): Mass-to-charge ratio (m/z) tolerance in ppm. Defaults to 15.
        """
        
        self.assay_links = None
        self.intensities = None
        self.rt_win = rt_win
        self.ppm = ppm
        self.feature_metadata = None
        self.structural_links = None


    def load_msi(self, df):
        """
        Imports MSI intensity data and extracts feature metadata from column names.

        Args:
            df (pandas.DataFrame): Data frame with MSI intensity data.
                - rows: samples
                - columns: features (m/z peaks)
                    Column names in the format:
                        <m/z>
                    For example:
                        149.111
        """

        _df = df.copy()

        # Extract column names
        names = _df.columns
        deconstructed = pd.DataFrame()
        for name in names:
            assay = name.split('_')[0]
            mz = float(name.split('_')[1])
            line = pd.DataFrame({
                'Feature': name,
                'Assay': assay,
                'RT': 1,
                'm/z': mz
            }, index=[0]
            )
            deconstructed = pd.concat([deconstructed, line])
        deconstructed.reset_index(inplace=True, drop=True)

        # Split metadata into assays
        data = []
        unique_assays = deconstructed['Assay'].unique()
        for assay in unique_assays:
            data.append(deconstructed[deconstructed['Assay'] == assay])

        # Save data
        self.feature_metadata = deconstructed
        self.assay_links = data
        self.intensities = _df


    def load_lcms(self, df):
        """
        Imports LC-MS intensity data and extracts feature metadata from column names.

        Args:
            df (pandas.DataFrame): Data frame with LC-MS intensity data.
                - rows: samples
                - columns: features (LC-MS peaks)
                    Column names in the format:
                        <Assay Name>_<RT in sec>_<m/z>m/z
                    For example:
                        HPOS_233.25_149.111m/z
        """

        _df = df.copy()

        # Extract column names
        _df_columns = _df.columns
        names = _df_columns.to_list()  # Extract names of features

        # Extract metadata from column names
        deconstructed = pd.DataFrame()
        for name in names:
            assay = name.split('_')[0]  
            rt = float(name.split('_')[1])  
            mz = name.split('_')[2]  
            mz = float(mz.split('m')[0])  
            line = pd.DataFrame({
                'Feature': name,
                'Assay': assay,
                'RT': rt,
                'm/z': mz
            }, index=[0]
            )
            deconstructed = pd.concat([deconstructed, line])
        deconstructed.reset_index(inplace=True, drop=True)

        # Split metadata into assays
        data = []
        unique_assays = deconstructed['Assay'].unique()
        for assay in unique_assays:
            data.append(deconstructed[deconstructed['Assay'] == assay])

        # Save data
        self.feature_metadata = deconstructed
        self.assay_links = data
        self.intensities = _df

    def get_structural_clusters(self, adducts='all', annotate=True):
        """
        Searches structural signatures in LC-MS data based on their m/z and RT. These structural signatures include 
        isotopologues and adduct patterns.

        Args:
            adducts (str, optional): Define what type of adducts to . 
                Possible values are:
                    - 'all': All adducts combinations (based on Fiehn Lab adduct calculator).
                    - 'most-common': Most common adducts for ESI (based on Waters documentation).
                Defaults to 'all'.
            annotate (bool, optional): Annotate significant features based on National Phenome Centre RIO data.
                Only to be run if the data was analysed by the National Phenome Centre or analysis followed their
                conventions and protocls. For more infomrmation see https://doi.org/10.1021/acs.analchem.6b01481 
                or https://phenomecentre.org.
                Uses semi-targeted annotations for selected compounds.
                Defaults to True.

        Returns:
            pandas.DataFrame or list(pandas.DataFrame): DataFrame of significant features with structural clusters.
        """

        # Check if metadata have been loaded
        if self.feature_metadata is None:
            warnings.simplefilter('error', RuntimeWarning)
            warnings.warn("No data loaded. Use 'load_lcms()' to load data.",
                          RuntimeWarning)

        # Get isotopologue and adduct clusters
        self._get_isotopologue_groups()
        self._get_adduct_groups()
        self._get_unified_struct_groups()

        # Get annotation from ROI files (NPC)
        if annotate:
            self._get_annotation()

        data = self.assay_links

        # PROCESS ASSAYS
        data_both = []
        iso_offset = 0
        adduct_offset = 0
        cluster_offset = 0
        for frame in data:

            working_frame = frame
 
            data_both.append(working_frame)


            # Update offsets for isotopologue and adduct clusters
            working_frame['Isotopologue group'] = working_frame['Isotopologue group'].apply(lambda x: x + iso_offset)
            working_frame['Adduct group'] = working_frame['Adduct group'].apply(lambda x: x + adduct_offset)
            working_frame['Structural cluster'] = working_frame['Structural cluster'].apply(lambda x: x + cluster_offset)
            iso_offset = working_frame['Isotopologue group'].max()  # Update offset
            adduct_offset = working_frame['Adduct group'].max()  # Update offset
            cluster_offset = working_frame['Structural cluster'].max()  # Update offset

        data_both = pd.DataFrame(np.vstack(data_both), columns=data_both[1].columns)
        self.structural_links = data_both
        # Get cross-assay links
        self._get_cross_assay_links()
        
        return  self.structural_links    

    def _get_isotopologue_groups(self):
        """
        Search for isotopologue signature in individual assay data frames.
        """

        for index, frame in enumerate(self.assay_links):
            # Create a copy of data frame
            frame = frame.copy()
            # Sort copied data frame and create new column
            frame.sort_values(by='m/z', inplace=True)
            frame.reset_index(inplace=True, drop=True)
            frame['Isotopologue group'] = [np.NaN] * len(frame)
            frame['Isotopologue pattern'] = [np.NaN] * len(frame)

            # Group ID for new cluster
            iso_group = 1
            m_plus = 0

            # Loop through data frame. Add neutron mass to current value
            for i in range(len(frame)):
                current_mz = frame.loc[i, 'm/z']
                current_mz += 1.003355
                current_rt = frame.loc[i, 'RT']
    
                
                # Loop through data frame to find expected MZ
                for j in range(len(frame)):
                    expected_mz = frame.loc[j, 'm/z']
                    expected_rt = frame.loc[j, 'RT']

                    # Get RT and PPM differences between the peaks
                    ppm_diff = self._mean_ppm_diff(expected_mz, current_mz)
                    rt_diff = abs(current_rt - expected_rt)

                    # Check if RT and PPM are within tolerance and add group ID
                    if ppm_diff <= self.ppm and rt_diff <= self.rt_win:
                        # If NaN use a new cluster ID
                        if np.isnan(frame.loc[i, 'Isotopologue group']):
                            frame.at[i, 'Isotopologue group'] = iso_group
                            frame.at[j, 'Isotopologue group'] = iso_group
                            iso_group += 1
                            m_plus = 0  # Reset M+ counter if new isotopologue group is found

                        # If NOT NaN use current cluster ID
                        if not np.isnan(frame.loc[i, 'Isotopologue group']):
                            frame.at[j, 'Isotopologue group'] = frame.loc[i, 'Isotopologue group']
                            # Assign M and M+ values
                            frame.at[i, 'Isotopologue pattern'] = m_plus
                            m_plus += 1
                            frame.at[j, 'Isotopologue pattern'] = m_plus
            
            # Update Current DataFrame
            self.assay_links[index] = frame

    def _get_adduct_groups(self, adducts='all'):
        """
        Search for adduct grouping signatures within significant features. The methods finds 

        Args:
            adducts (str, optional): _description_. Defaults to 'all'.
        """

        for index, frame in enumerate(self.assay_links):

            frame_ = frame.copy()

            # Detect isotopologues within one loop

            # Get neutral masses for all adducts
            frame__ = self.get_neutral_mass(features=frame_, adducts=adducts)

            # Search for adducts in current DataFrame
            data_clusters_frame = self._search_main_adduct(frame__)

            # Combine isotopologue and adduct clusters into one DataFrame
            frame_2 = frame__.iloc[:, :6]
            working_frame = frame_2.merge(data_clusters_frame.iloc[:, [0, 7, 2, 3, 4, 5]], on='Feature', how='left')

            # Merge overlapping adduct clusters

            non_unique_features = working_frame['Feature'][working_frame['Feature'].duplicated(keep=False)].unique()
            for item in non_unique_features:
                # For all non-unique features, find all clusters they belong too
                fr_ = working_frame[working_frame['Feature'] == item].loc[:, ['Feature', 'Adduct group', 'Adduct']]
                combined_adduct = '/'.join(fr_['Adduct'])
                fr_['Adduct'] = combined_adduct

                #
                working_frame.reset_index(inplace=True, drop=True)
                fr_.reset_index(inplace=True, drop=True)
                for i in range(len(fr_)):
                    working_frame.loc[working_frame['Feature'] ==
                                      fr_.loc[i, 'Feature'], 'Adduct'] = fr_.loc[i, 'Adduct']

                # unify all overlapping cluster by assigning the lowest cluster values to all clusters
                for i in range(len(fr_) - 1):
                    working_frame['Adduct group'].replace({fr_.iloc[i + 1, 1]: fr_.iloc[0, 1]}, inplace=True)
                    # working_frame.reset_index(inplace=True, drop=True)
                    # working_frame['Adduct'].replace({fr_.iloc[i, 2]: fr_.iloc[0, 2]}, inplace=True)

                working_frame = working_frame.drop_duplicates(subset='Feature')  # Delete non unique Features

            self.assay_links[index] = working_frame
            # now load data below in the main loop as nothing is returned

    def get_neutral_mass(self, features, adducts='all'):
        """
        Calculate potential neutral masses for all m/z features.

        Args:
            features (pandas.DataFrame): DataFrame containing m/z features.
            adducts (str, optional): DataFrame with ion masses and names. Defaults to 'all'.

        Returns:
            pandas.DataFrame: DataFrame with m/z and hypothetical neutral masses for given adducts.
        """
        
        # Load all files with "all" adducts
        if adducts == 'all':
            stream_all_adducts_pos = pkg_resources.resource_stream(__name__, 'Data/Adducts/all_adducts_pos.csv')

            adducts_positive = pd.read_csv(stream_all_adducts_pos)
            
            stream_all_adducts_neg = pkg_resources.resource_stream(__name__, 'Data/Adducts/all_adducts_neg.csv')
            adducts_negative = pd.read_csv(stream_all_adducts_neg)

        # Load files with the "most common" adducts
        else:
            # Load external adduct files
            stream_common_adducts_pos = pkg_resources.resource_stream(__name__, 'Data/Adducts/common_adducts_pos.csv')
            adducts_positive = pd.read_csv(stream_common_adducts_pos)
            
            stream_common_adducts_neg = pkg_resources.resource_stream(__name__, 'Data/Adducts/common_adducts_neg.csv')
            adducts_negative = pd.read_csv(stream_common_adducts_neg)

        df = features.copy()  # Copy features data frame
        df.reset_index(inplace=True, drop=True)
        assay = df.loc[0, 'Assay']
        if re.search(r'pos', assay,  re.IGNORECASE):
            adducts = adducts_positive
        elif re.search(r'neg', assay,  re.IGNORECASE):
            adducts = adducts_negative
        else:
            raise Exception("Ionisation mode has not been recognised. Please ensure that ionisation mode is "
                            "either 'POS' for positive or 'NEG' for negative types of ionisation")

        adducts = adducts.copy()  # Copy adduct ion masses
        adducts.reset_index(inplace=True, drop=True)
        for j in range(adducts.shape[0]):  # Calculate hypothetical neutral masses for all given adduct ...
            column = []
            for i in range(len(df)):  # ... and repeat for all peaks
                column.append((df.loc[i, 'm/z'] - adducts.loc[j, 'Mass']) / adducts.loc[j, '1/Charge'])  # Calculate U
            name = adducts.loc[j, 'Ion name']  # Get the column name from adduct name in the adduct file
            df.insert(features.shape[1] + j, name, column)  # Append column of neutral masses for given adduct to the DF
        return df

    def _search_main_adduct(self, x):
        """
        Search for main adducts ([M+H]+ / [M-H]-) in the given input.

        Args:
            x (DataFrame): The input DataFrame containing the data to search.

        Returns:
            DataFrame: A DataFrame containing the matches found for the main adducts.
        """
        
        frame_ = x.copy()

        # Outer Loop - go through main adducts
        matches = pd.DataFrame()
        cluster_flag = 1
        for index, row in frame_.iterrows():
            group = self._find_adduct_matches(frame_, row, cluster_flag)
            if len(group) > 0:
                matches = pd.concat([matches, group])
                cluster_flag = cluster_flag + 1
            else:
                matches = pd.concat([matches, group])

        return matches

    def _find_adduct_matches(self, frame_, row_, clust_flag, main_adduct=True):
        """
        Method searches for adducts within a given ppm window.

        Args:
            frame_ (pandas.DataFrame): DataFrame containing m/z values.
            row_ (pandas.Series): Row of the DataFrame.
            clust_flag (bool): Cluster flag.
            main_adduct (bool): Flag for main adducts.

        Returns:
            pandas.DataFrame: DataFrame with adducts.
        """

        main_adduct_index = 6  # This is a default column, needs to be updated if default adduct changes

        frame = frame_.copy()
        group = pd.DataFrame(columns=['Feature', 'm/z', 'Expected neutral mass',
                                      'Observed neutral mass', 'Neutral mass |difference ppm|',
                                      'Adduct', 'RT', 'Adduct group'
                                      ])  # Add groups one by one and see what the result looks like

        frame = frame[~(frame['Feature'] == row_['Feature'])]  # Delete current row from frame
        rt_now = row_['RT']  # Filter frame within RT window
        frame = frame.loc[(frame['RT'] >= rt_now - self.rt_win) & (frame['RT'] <= rt_now + self.rt_win)]  # RT WINDOW
        if row_['Isotopologue group'] > 0:  # Filter out all suspected isotopologues identical to current row
            frame = frame[~(frame['Isotopologue group'] == row_['Isotopologue group'])]
        val_current = row_[main_adduct_index]

        # Loop through what remained from frame
        for i in range(frame.shape[0]):  # Search in rows
            for j in range(main_adduct_index, frame.shape[1]):  # Search in columns  MUST BE UPDATED FOR FLEXIBILITY OF COLUMNS!
                ppm_diff1 = self._mean_ppm_diff(frame.iloc[i, j], val_current)
                if ppm_diff1 <= self.ppm:
                    line = pd.DataFrame({
                        'Feature': frame.iloc[i, 0],
                        'm/z': frame.iloc[i, 3],
                        'Expected neutral mass': val_current,
                        'Observed neutral mass': frame.iloc[i, j],
                        'Neutral mass |difference ppm|': ppm_diff1,
                        'Adduct': frame.columns[j],
                        'RT': frame.iloc[i, 2],
                        'Adduct group': clust_flag
                    }, index=[0])
                    group = pd.concat([group, line])

        # If there are matches, add the current row into the group
        if len(group) > 0:
            line = pd.DataFrame({
                'Feature': row_['Feature'],
                'm/z': row_['m/z'],
                'Expected neutral mass': val_current,
                'Observed neutral mass': val_current,
                'Neutral mass |difference ppm|': 0,
                'Adduct': frame.columns[6],  # This is a default column, needs to be updated if default adduct changes
                'RT': row_['RT'],
                'Adduct group': clust_flag
            }, index=[0])

            group = pd.concat([group, line])

        return group

    def _get_unified_struct_groups(self):
        """
        Unifies 'Isotopologue group' and 'Adduct group' into a single 'Structural group'.
        """

        for index, frame in enumerate(self.assay_links):

            frame_ = frame.copy()  # Copy the input dataframe

            # Stack Isotopologue and adduct clusters into a single variable 'Structural cluster'
            offset = frame_['Isotopologue group'].max()  # Get the highest number of iso group
            frame_['Adduct group'] = frame_['Adduct group'].apply(lambda x: x+offset)  # adduct clusters above iso
            adduct_frame = frame_.dropna(subset=['Adduct group'])  # Get only rows with non 0 clusters
            iso_frame = frame_.dropna(subset=['Isotopologue group'])  # Get only rows with non 0 clusters
            adduct_frame.insert(0, 'Structural cluster', adduct_frame['Adduct group'])  # Create structural cluster col
            iso_frame.insert(0, 'Structural cluster', iso_frame['Isotopologue group'])  # Create structural cluster col
            adduct_frame = adduct_frame.loc[:, ['Feature', 'Structural cluster']]
            iso_frame = iso_frame.loc[:, ['Feature', 'Structural cluster']]
            stacked_frames = pd.DataFrame(np.vstack([adduct_frame, iso_frame]), columns=iso_frame.columns)
            frame_ = frame_.merge(stacked_frames, on='Feature', how='outer')

            # Delete non-unique values
            non_unique_features = frame_['Feature'][frame_['Feature'].duplicated(keep=False)].unique()
            for item in non_unique_features:
                # For all non-unique features, find all clusters they belong too
                fr_ = frame_[frame_['Feature'] == item].loc[:, ['Feature', 'Structural cluster']]
                # unify all overlapping cluster by assigning the lowest cluster values to all clusters
                for i in range(len(fr_) - 1):
                    frame_['Structural cluster'].replace({fr_.iloc[i + 1, 1]: fr_.iloc[0, 1]}, inplace=True)
            unified_frame = frame_.drop_duplicates(subset='Feature')  # Delete non unique Features

            # Update the current assay metadata with the unified structural clusters
            self.assay_links[index] = unified_frame


    def _get_annotation(self, roi=None):
        """
        Annotates hypothetical features using the RIO files from the National Phenome Centre.

        Args:
            roi (pandas.DataFrame, optional): Region of Interest (RIO) file from the National Phenome Centre. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame with structural clusters and feature annotations.
        """

        # Load RIO files
        for index, frame in enumerate(self.assay_links):

            frame = frame.copy()
                
            if roi is None:
                if frame.loc[0, 'Assay'] == 'HPOS':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/HPOS_ROI_V_3_2_1.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'UHPOS':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/HPOS_ROI_V_3_2_1.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'SHPOS':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/HPOS_ROI_V_3_2_1.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'LPOS':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/LPOS_ROI_V_5_1_2.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'LNEG':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/LNEG_ROI_V_5_1_1.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'RPOS':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/RPOS_ROI_V_3_2_0.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                elif frame.loc[0, 'Assay'] == 'RNEG':
                    roi_stream = pkg_resources.resource_stream(__name__, 'Data/ROI/RNEG_ROI_V_3_2_0.csv')
                    roi = pd.read_csv(roi_stream, encoding='windows-1252')

                else:
                    raise Exception("ANNOTATION ERROR - Assay has not been recognised. Please ensure that assay name "
                                    "matches naming convention of the National Phenome Centre (NPC) or run structural "
                                    "search without annotation. For more information on NPC assay naming convention please "
                                    "visit https://github.com/phenomecentre/npc-open-lcms")

            else:
                roi = roi

            # Create copies of = DataFrames
            frame_ = frame.copy()
            frame_.reset_index(inplace=True, drop=True)
            roi_ = roi.copy()
            roi_.reset_index(inplace=True, drop=True)
            # Create empty dataframe for annotations
            _annotations = pd.DataFrame(columns=['cpdName', 'chemicalFormula', 'ion', 'mzMin', 'mzMax', 'rtMin', 'rtMax'])
            # Iterate through all features
            for i in range(len(frame_)):
                id_current = frame_.loc[i, 'Feature']
                rt_current = frame_.loc[i, 'RT']  # variable used in the query
                mz_current = frame_.loc[i, 'm/z']  # variable used in query
                query = "mzMin < @mz_current < mzMax & rtMin < @rt_current < rtMax"  # query to search the dataframe
                retrieved = roi_.query(query, inplace=False)  # query the dataframe
                retrieved.reset_index(inplace=True, drop=True)  # reindex allows to use .loc[]
                if retrieved.shape[0] > 0:  # If any annotations found
                    line = pd.DataFrame({
                        'Feature': id_current,
                        'cpdName': retrieved.loc[0, 'cpdName'],
                        'chemicalFormula': retrieved.loc[0, 'chemicalFormula'],
                        'ion': retrieved.loc[0, 'ion'],
                        'mzMin': retrieved.loc[0, 'mzMin'],
                        'mzMax': retrieved.loc[0, 'mzMax'],
                        'rtMin': retrieved.loc[0, 'rtMin'],
                        'rtMax': retrieved.loc[0, 'rtMax']
                    }, index=[0])
                    _annotations = pd.concat([_annotations, line])

            # If any annotation found, then save them merge them with original dataframe
            if _annotations.shape[0] > 0:
                frame_annotation = frame_.merge(_annotations, on='Feature', how='left')
            else:
                # If no annotations were found, then add to the frame_ empty columns from _annotations
                frame_annotation = frame_.reindex(frame_.columns.union(_annotations.columns), axis=1)
                frame_annotation = frame_annotation.reindex(frame_.columns.tolist() + _annotations.columns.tolist(), axis=1)

            self.assay_links[index] = frame_annotation
            # return frame_annotation

    def _get_cross_assay_links(self):
        """
        Search for cross-assay links between features based on their m/z and RT. 
        Features that are part of sturctural cluster are linked only on their M+H and M-H adducts.
        For singletons nutral mass estimated from M+H and M-H is used for cross-assay linking.
        """
        
        # Load datea
        struct_data = self.structural_links
        signletons_data = self.structural_links
        data_original = self.structural_links

        # Adduct groups
        # selecto only the rows where the structural cluster is not NaN
        struct_data = struct_data.dropna(subset=['Structural cluster'])
        # get only rows where adduct is like M+H or M-H
        struct_data = struct_data[struct_data['Adduct'].str.contains(r'\[M\+H\]|\[M-H\]', regex=True, na=False)]
        # delete duplicates of Expected neutral mass
        struct_data = struct_data.drop_duplicates(subset='Expected neutral mass')
        struct_data.reset_index(inplace=True, drop=True)

        # Singletons and Isoptopologues
        # calculate expected neutral mass for singletons
        signletons_data = signletons_data.copy()
        # frome signletons_data get only rows where the structural cluster is NaN
        signletons_data = signletons_data[signletons_data['Expected neutral mass'].isna()]
        # For column 'Isotopologue group' keep only rows where the values is 0 or NaN
        signletons_data = signletons_data[((signletons_data['Isotopologue pattern'] == 0) 
                                         | (pd.isna(signletons_data['Isotopologue pattern'])))]
        signletons_data.reset_index(inplace=True, drop=True)
        # if assay is like POS then add 1.007276466812 to the m/z value
        expected = signletons_data.apply(lambda x: x['m/z'] - 1.007276466812 if re.search(r'POS', x['Assay'], re.IGNORECASE) 
                                         else x['m/z'] + 1.007276466812, axis=1)
        expected.reset_index(inplace=True, drop=True)
        signletons_data.loc[:, 'Expected neutral mass'] = expected

        # Merge Adduct and Singletons data
        data = pd.concat([struct_data, signletons_data])
        data.reset_index(inplace=True, drop=True)

        # Create an empty column and isotopolouge group flag
        data['Cross-assay link'] = [np.NaN] * data.shape[0]
        cross_assay_flag = 1

        # loop through all rows and check if the expcted neutral mass for i the same as for j
        for i in range(len(data)):
            for j in range(len(data)):

                # check if the Assay is different
                if data.iloc[i, 1] != data.iloc[j, 1]:
                    PPM_diff = self._mean_ppm_diff(data.loc[i, 'Expected neutral mass'], data.loc[j, 'Expected neutral mass'])
                    RT_diff = abs(data.loc[i, 'RT'] - data.loc[j, 'RT'])

                    # check if the two is LPOS and LNEG or RPOS and RNEG
                    if ((re.search(r'LPOS', data.iloc[i, 1], re.IGNORECASE) and
                        re.search(r'LNEG', data.iloc[j, 1], re.IGNORECASE)) or
                        (re.search(r'RPOS', data.iloc[i, 1], re.IGNORECASE) and 
                         re.search(r'RNEG', data.iloc[j, 1], re.IGNORECASE))
                    ):
                        if PPM_diff <= self.ppm and RT_diff <= self.rt_win:
                            if np.isnan(data.loc[i, 'Cross-assay link']) and np.isnan(data.loc[j, 'Cross-assay link']):
                                data.loc[i, 'Cross-assay link'] = cross_assay_flag
                                data.loc[j, 'Cross-assay link'] = cross_assay_flag
                                cross_assay_flag += 1
                            elif np.isnan(data.loc[i, 'Cross-assay link']):
                                    data.loc[i, 'Cross-assay link'] = data.loc[j, 'Cross-assay link']
                            elif np.isnan(data.loc[j, 'Cross-assay link']):
                                    data.loc[j, 'Cross-assay link'] = data.loc[i, 'Cross-assay link']
                    
                    # If for pair with different LC
                    else:
                        if PPM_diff <= self.ppm:
                            if np.isnan(data.loc[i, 'Cross-assay link']) and np.isnan(data.loc[j, 'Cross-assay link']):
                                data.loc[i, 'Cross-assay link'] = cross_assay_flag
                                data.loc[j, 'Cross-assay link'] = cross_assay_flag
                                cross_assay_flag += 1
                            elif np.isnan(data.loc[i, 'Cross-assay link']):
                                    data.loc[i, 'Cross-assay link'] = data.loc[j, 'Cross-assay link']
                            elif np.isnan(data.loc[j, 'Cross-assay link']):
                                    data.loc[j, 'Cross-assay link'] = data.loc[i, 'Cross-assay link']

        # create new dataframe that only contains 'Feature' and 'Cross-assay link'
        cross_assay_links = data.loc[:, ['Feature', 'Cross-assay link']]

        # add cross-assay links to the original dataframe
        data = data_original.merge(cross_assay_links, on='Feature', how='left')

        self.structural_links = data

    @staticmethod
    def _mean_ppm_diff(x, y):
        """
        Calculates the mean PPM difference between two numbers.

        Args:
            x (float): The first number.
            y (float): The second number.

        Returns:
            float: Mean PPM difference between `x` and `y`.
        """

        diff1 = abs((x - y) / x * 1000000)
        diff2 = abs((y - x) / y * 1000000)
        return (diff1 + diff2) / 2
    
    def get_correlation_clusters(self, flat_method ='constant', cut_threshold=0.7, max_clusters=5, cor_method='pearson' ,
                                 linkage_method='complete', metric='euclidian', **kwargs):
        """
        Clusters features based on their correlation. The method uses hierarchical clustering to create clusters.

        Args:
            flat_method (str {'constant', 'silhouette'}, optional):
                Method for cluster flattening:
                - 'constant': Flattens clusters based on a constant threshold (cut_threshold).
                - 'silhouette': Flattens clusters based on most optimal silhouette score.
                Defaults to 'constant'.
            cut_threshold (float, optional): Constant threshold for flattening clusters. Defaults to 0.7.
            max_clusters (int, optional): Maximum number of clusters for silhouete method. Defaults to 5.
            cor_method (str {'pearson', 'kendall', 'spearman'}, optional): Mehthod for calculation correlations. Defaults to 'pearson'.
            linkage_method (str, optional): Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation.
                The algorithm will merge the pairs of cluster that minimize this criterion.
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
            metric (str, optional): The distance metric to use. The metric to use when calculating distance between instances in a feature array.
                Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
                If linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix is needed as input for the fit method.
                Defaults to 'euclidian'.
        """
       
        # Check if data were loaded
        if self.feature_metadata is None:
            warnings.simplefilter('error', RuntimeWarning)
            warnings.warn("No data loaded. Use 'load_lcms()' to load data.",
                          RuntimeWarning)

        # Validate the input method
        # List of acceptable arguments
        acceptable_flat_methods = ['constant', 'silhouette']

        if flat_method not in acceptable_flat_methods:
            raise ValueError(f"Invalid flatting method '{flat_method}'. Choose from {acceptable_flat_methods}.")

        # Calculate correaltion and sissimilarity
        df = self.intensities
        # Calculate correlation between variables
        correlation = df.corr(method=cor_method)
        # Calculate dissimilarity
        dissimilarity = 1 - abs(correlation)

        if flat_method == 'constant':
        # Check if metadata have been loaded
            z = linkage(squareform(dissimilarity), method=linkage_method, metric=metric)
            # Get flat clusters
            f_clust = fcluster(z, t=cut_threshold, criterion='distance', depth=100)
            correlation_clusters = pd.DataFrame({ 'Feature': df.columns, 'Correlation cluster':f_clust})

            # Plot correlation heatmap and dendrogram
            plt.subplots(figsize=(17, 10))
            sns.heatmap( data=correlation, annot=False, annot_kws={"fontsize":6})
            plt.show()
            plt.figure(figsize=(17.3, 5))
            dendrogram( Z=z, labels=df.columns, orientation='top', leaf_rotation=90, color_threshold=cut_threshold, **kwargs)
            plt.show()
            
            # Save data
            self.correlation_clusters = correlation_clusters
            self.structural_links['Correlation cluster'] = f_clust

        elif flat_method == 'silhouette':

            # Replace 'max_clusters' with the maximum number of clusters you want to consider
            cluster_range = range(2, max_clusters + 1)
            # Initialize an empty dictionary to store silhouette scores for each number of clusters
            silhouette_scores = {}
            # Iterate over the range of cluster numbers
            for n_clusters in cluster_range:
                # Create an AgglomerativeClustering model with 'n_clusters'
                model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage_method)

                # Fit the model to the dissimilarity matrix
                cluster_labels = model.fit_predict(dissimilarity)

                # Calculate the silhouette score for the current number of clusters
                silhouette_scores[n_clusters] = silhouette_score(dissimilarity, cluster_labels, metric='precomputed')

            # Find the number of clusters with the highest silhouette score
            best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
            best_score = silhouette_scores[best_n_clusters]
            # Print the best number of clusters and its silhouette score
            print(f"Best number of clusters based on silhouette score: {best_n_clusters}")
            print(f"Silhouette score for {best_n_clusters} clusters: {best_score}")
            # Now you can use 'best_n_clusters' to create flat clusters
            model = AgglomerativeClustering(n_clusters=best_n_clusters, metric='precomputed', linkage=linkage_method)
            flat_clusters = model.fit_predict(dissimilarity)

            # Save data
            self.correlation_clusters = flat_clusters
            self.structural_links['Correlation cluster'] = flat_clusters


            # Visualisation
            plt.subplots(figsize=(17, 10))
            sns.heatmap( data=correlation, annot=False, annot_kws={"fontsize":6})
            plt.show()

            # Compute the silhouette scores for each sample
            silhouette_vals = silhouette_samples(dissimilarity, flat_clusters, metric='precomputed')
            # Compute the mean silhouette score
            silhouette_avg = silhouette_score(dissimilarity, flat_clusters, metric='precomputed')
            # Plotting the silhouette plot
            fig, ax = plt.subplots()
            y_lower = 10
            for i in range(best_n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i
                ith_cluster_silhouette_vals = silhouette_vals[flat_clusters == i]
                ith_cluster_silhouette_vals.sort()

                size_cluster_i = ith_cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.nipy_spectral(float(i) / best_n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax.set_title("The silhouette plot for best number of clusters.")
            ax.set_xlabel("The silhouette coefficient values")
            ax.set_ylabel("Cluster label")
            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks(np.arange(-0.1, 1.1, 0.2))
            plt.show()


    def get_structural_network(self, include_all=False, interactive=False, 
                               return_nx_object=False,  output_file='interactive.html', labels=False, master_file=None):
        """
        Generates a structural network graph based on the provided master file or the loaded structural links data.

        Args:
            include_all (bool, optional): Whether to include all features in the network, even if they are not structurally linked to other features.
                Defaults to False.
            interactive (bool, optional): Whether to display the network graph interactively using pyvis.network.
                If False, the network graph is displayed using NetworkX and Matplotlib.
                Defaults to False.
            return_nx_object (bool, optional): Whether to return the NetworkX object representing the network graph edited in CytoScape. 
                Defaults to False.
            output_file (str, optional): The name of the output file when displaying the network graph interactively using pyvis.network.
                Only applicable when interactive is True. 
                Defaults to 'interactive.html'.
            labels (bool, optional): Whether to display labels for the nodes in the network graph.
                Only applicable when interactive is False. 
                Defaults to False.
            master_file (pd.DataFrame, optional): The master file containing necessary columns for generating the network.
                This is intended for cases when strucutural links required manual curation (e.g. manualy assigned isotopologue groups, adduct groups, etc.)
                If not provided, the function uses the loaded structural links data.
                Required columns: 
                    - Feature: Feature ID (e.g. HPOS_233.25_149.111m/z)
                    - Assay: Assay name (e.g. HPOS)
                    - Isotopologue group (groups features with similar isotopologue patterns)
                    - Isotopologue pattern (e.g. 0, 1, 2 ... N representing M+0, M+1, M+2 ... M+N)
                    - Adduct group (groups features with similar adduct patterns)
                    - Adduct (adduct label, e.g. [M+H]+, [M-H]-)
                    - Structural cluster (groups features with similar isotopologue and adduct patterns)
                    - Correlation cluster (flattedned hierarchical cluster from get_correlation_clusters()
                    - Cross-assay link (links features across different assays)
                    - cpdName (compound name, optional)
                Defaults to None.

        Returns:
            NetworkX.Graph or None: The NetworkX object representing the network graph, if return_nx_object is True.
                Edge weights represent the type of link between features:
                    - Isotopologue: 1
                    - Adduct: 5
                    - Cross-assay link: 10
                Otherwise, None.
                                    

        Raises:
            RuntimeWarning: If no data is loaded and no master file is provided.
            RuntimeWarning: If the provided master file is missing necessary columns.

        Notes:
            - The function creates a network graph based on the provided master file or the loaded structural links data.
            - The network graph includes nodes representing features and edges representing different types of links.
            - The graph can be displayed interactively using pyvis.network or using networkx and matplotlib.
            - The graph can be saved as a NetworkX object if return_nx_object is True.
        """
        
        # Check if data have been loaded
        if master_file is None and self.structural_links is None:
            warnings.simplefilter('error', RuntimeWarning)
            warnings.warn("No data loaded. Use 'get_structural_clusters()' to create structural links file or load a " 
                          "master file using the 'master_file' parameter.",
                          RuntimeWarning)
        
        # Import master if 
        if master_file is not None:
            # Define necessary colums for generating the network
            required_columns = ['Feature',
                                 'Assay',
                                 'Isotopologue group',
                                 'Isotopologue pattern',
                                 'Adduct group',
                                 'Adduct',
                                 'Structural cluster',
                                 'Correlation cluster', 
                                 'Cross-assay link']
            missing_columns = [col for col in required_columns if col not in master_file.columns]
            if not missing_columns:
                master = master_file.copy()
            else:
                warnings.simplefilter('error', RuntimeWarning)
                warnings.warn(f"Missing columns in the master file: {missing_columns}",
                          RuntimeWarning)
        else:
            master = self.structural_links.copy()

        # Drop all features that are not structurally linked to any other feature
        if not include_all:
            master.dropna(subset=['Structural cluster', 'Cross-assay link'], how='all', inplace=True)

       # Optional Columns for the network
        if 'cpdName' not in master.columns:
            master['cpdName'] = np.nan 

        master['Node'] = master['Feature']  # Create a node column

        # Create a network graph
        G = nx.Graph()
        for index, row in master.iterrows():
            G.add_node(row['Node'], 
                       Assay=row['Assay'], 
                       Isotopologue_group=row['Isotopologue group'],
                       Isotopologue_pattern=row['Isotopologue pattern'], 
                       Adduct_group=row['Adduct group'],
                       Adduct=row['Adduct'], 
                       Structural_cluster=row['Structural cluster'],
                       Correlation_cluster=row['Correlation cluster'], 
                       Cross_assay_link=row['Cross-assay link'],
                       Annotation=row['cpdName']
                       )

        # Add edges to the graph
        for u in G.nodes():
            for v in G.nodes():
                if (
                    u != v and G.nodes[u]['Structural_cluster'] == G.nodes[v]['Structural_cluster'] and 
                    ( G.nodes[u]['Adduct_group'] == G.nodes[v]['Adduct_group'] )
                ):
                    G.add_edge(u, v, weight = 5, type= 5)  # Adduct weight = 5

                if u != v and G.nodes[u]['Cross_assay_link'] == G.nodes[v]['Cross_assay_link']:
                    G.add_edge(u, v, weight = 10, type= 10)  # Cross assay link weight = 10

                if (u != v and 
                    G.nodes[u]['Structural_cluster'] == G.nodes[v]['Structural_cluster'] and 
                    (G.nodes[u]['Isotopologue_group'] == G.nodes[v]['Isotopologue_group'] and 
                    G.nodes[u]['Isotopologue_pattern'] == G.nodes[v]['Isotopologue_pattern']+1 )
                ):
                    G.add_edge(u, v, weight = 1, type= 1)  # Isotopologue weight = 1

        # Define colours
        correlation_clusters = set(nx.get_node_attributes(G, 'Correlation_cluster').values())
        colour_map = {cluster: plt.cm.tab10(i) for i, cluster in enumerate(correlation_clusters)}

        # Chose plot option btween networkx and pyvis.network 
        if  interactive:
            iG = G.copy()
            # create vis network
            net = Network(notebook=True, cdn_resources='remote')
            # load the networkx graph
            net.from_nx(iG)
            net.show(output_file)
            display(IFrame(output_file, width="100%", height="600px"))

        else:
            pos = nx.spring_layout(G,  threshold=0.015)
            node_colors = [colour_map[G.nodes[node]['Correlation_cluster']] for node in G.nodes()]
            nx.draw(G, pos, with_labels=labels, font_size=6, node_color=node_colors, node_size=300, edge_color='gray')
            plt.title("Structural Network")
            plt.show()
        
        # Save NetworkX object
        if return_nx_object:
            return G
