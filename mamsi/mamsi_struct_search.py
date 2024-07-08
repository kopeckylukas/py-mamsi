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
import os
import warnings
import matplotlib.pyplot as plt
import pkg_resources
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns

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
        get_structural_clusters(adducts='all', annotate=True, return_as_single_frame=False): Searches structural
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

    def get_structural_clusters(self, adducts='all', annotate=True, return_as_single_frame=False):
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
                Uses semi-targeted annotations for selected compounds
                Should be used only with assays analysed by the National Phenome Centre. For more information 
                visit National Phenome Centre's website: https://phenomecentre.org.
                Defaults to True.
            return_as_single_frame (bool, optional): Option to return all significant features in a single DataFrame. 
                Options are:
                    - True: Return all features in a single DataFrame.
                    - False: Return all features in a list of DataFrames.
                Defaults to False.

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
            print('Annotation done')


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
            if return_as_single_frame:
                working_frame['Isotopologue group'] = working_frame['Isotopologue group'].apply(lambda x: x + iso_offset)
                working_frame['Adduct group'] = working_frame['Adduct group'].apply(lambda x: x + adduct_offset)
                working_frame['Structural cluster'] = working_frame['Structural cluster'].apply(lambda x: x + cluster_offset)
                iso_offset = working_frame['Isotopologue group'].max()  # Update offset
                adduct_offset = working_frame['Adduct group'].max()  # Update offset
                cluster_offset = working_frame['Structural cluster'].max()  # Update offset

        # Return data as a single frame
        if return_as_single_frame:
            data_both = pd.DataFrame(np.vstack(data_both), columns=data_both[1].columns)
        
        self.structural_links = data_both

        return data_both    

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
            frame['M+'] = [np.NaN] * len(frame)

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
                            frame.at[i, 'M+'] = m_plus
                            m_plus += 1
                            frame.at[j, 'M+'] = m_plus
            
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
            # print(group.shape)
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
                                      'Adduct', 'RT', 'Adduct group', 'M+'
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
                'Adduct': frame.columns[5],  # This is a default column, needs to be updated if default adduct changes
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
    
    def get_correlation_clusters(self, get_data=True, get_charts=True, chart_anot=False,corr_method='pearson',
                             linkage_method='complete', linkage_metric='euclidian', cluster_threshold=0.7,
                             max_depth=100, criterion='distance', truncate_mode=None, p=30, **kwargs):
       

        # Check if metadata have been loaded
        if self.feature_metadata is None:
            warnings.simplefilter('error', RuntimeWarning)
            warnings.warn("No data loaded. Use 'load_lcms()' to load data.",
                          RuntimeWarning)
        
        df = self.intensities
        # Calculate correlation between variables
        correlation = df.corr(method=corr_method)
        # Calculate dissimilarity
        dissimilarity = 1 - abs(correlation)
        z = linkage(squareform(dissimilarity), method=linkage_method, metric=linkage_metric)
        # Get flat clusters
        f_clust = fcluster(z, t=cluster_threshold, criterion=criterion, depth=max_depth)
        correlation_clusters = pd.DataFrame({ 'Feature': df.columns, 'Correlation cluster':f_clust})

        # Plot correlation heatmap and dendrogram
        if get_charts:
            plt.subplots(figsize=(17, 10))
            sns.heatmap( data=correlation, annot=chart_anot, annot_kws={"fontsize":6})
            plt.show()
            plt.figure(figsize=(17.3, 5))
            dendrogram( Z=z, labels=df.columns, orientation='top', leaf_rotation=90, truncate_mode=truncate_mode, p=p, color_threshold=cluster_threshold, **kwargs)
            plt.show()

        self.correlation_clusters = correlation_clusters
        # return flat clusters
        if get_data: return correlation_clusters






