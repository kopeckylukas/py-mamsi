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

plt.rc('font', family='Verdana')


class MamsiStructSearch:

    def __init__(self, rt_win=5, ppm=15):
        self.assay_metadata = None
        self.intensities = None
        self.rt_win = rt_win
        self.ppm = ppm
        self.feature_metadata = None

    def load_lcms(self, df):
        """
        The method imports LC-MS intensity data and extracts feature metadata from column names.
        :param df: Pandas DataFrame
            Data frame with lc-ms intensity data.

            Column names in format
                <Assay Name>_<RT in sec>_<m/z>m/z
                e.g.
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
        self.assay_metadata = data
        self.intensities = np.array(_df)

    def get_correlation_clusters(self, visualise=True):

        # Check if metadata have been loaded
        if self.feature_metadata is None:
            warnings.simplefilter('error', RuntimeWarning)
            warnings.warn("No data loaded. Use 'load_lcms()' to load data.",
                          RuntimeWarning)
        pass

    def _get_adduct_groups(self, adducts='all'):


        for index, frame in enumerate(self.assay_metadata):

            frame_ = frame.copy()

            # Detect isotopologues within one loop

            # Get neutral masses for all adducts
            frame__ = self.get_neutral_mass(features=frame_, adducts=adducts)

            # Search for adducts in current dataframe
            data_clusters_frame = self._search_main_adduct(frame__)

            # Combine isotopologue and adduct clusters into one dataframe
            frame_2 = frame__.iloc[:, :5]
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

            self.assay_metadata[index] = working_frame
            # now load data below in the main loop as nothing is returned


    def get_structural_clusters(self, detect_iso=True, adducts='all', unify_overlap=True, annotate=True,
                                unify_clusters=True, return_as_single_frame=False):

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


        data = self.assay_metadata

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

        return data_both

    def get_neutral_mass(self, features, adducts='all'):
        """
        Method calculates potential neutral masses for all m/z features
        :param features: obejct
            dataframe with m/z
        :param adducts: str (default 'all')
            dataframe with ion masses and names
        :return: dataframe with m/z and hypothetical neutral masses for given adducts
        """
        
        # Load all filess with "all" adducts
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
                column.append((df.loc[i, 'm/z'] - adducts.loc[j, 'Mass']) / adducts.loc[j, '1/Charge'])  # Calculate mass
            name = adducts.loc[j, 'Ion name']  # Get the column name from adduct name in the adduct file
            df.insert(features.shape[1] + j, name, column)  # Append column of neutral masses for given adduct to the DF
        return df

    def _search_main_adduct(self, x):
        
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
        '''
        Method searches for adducts within a given ppm window
        :param frame_: object
            dataframe with m/z
        :param row_: object
            row of the dataframe
        :param clust_flag: cluster flag
        :param main_adduct: boolean flag for main adducts
        :return: dataframe with adducts
        '''
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
        val_current = row_[5]

        # Loop through what remained from frame
        for i in range(frame.shape[0]):  # Search in rows
            for j in range(5, frame.shape[1]):  # Search in columns  MUST BE UPDATED FOR FLEXIBILITY OF COLUMNS!
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
        Metdhod unifies 'Isotopologue group' and 'Adduct group' into single 'Structural group'
        """
        for index, frame in enumerate(self.assay_metadata):

            frame_ = frame.copy()  # Copy the input dataframe

            # Stack Isotopologue and adduct clusters into a single variable 'Struc
            offset = frame_['Isotopologue group'].max()  # Get the highest number of iso group
            frame_['Adduct group'] = frame_['Adduct group'].apply(lambda x: x+offset)  # adduct clusters above iso
            adduct_frame = frame_.dropna(subset=['Adduct group'])  # Get only rows with non 0 clusters
            iso_frame = frame_.dropna(subset=['Isotopologue group'])  # Get only rows with non 0 clusters
            adduct_frame.insert(0, 'Structural cluster', adduct_frame['Adduct group'])  # Create structural cluster column
            iso_frame.insert(0, 'Structural cluster', iso_frame['Isotopologue group'])  # Create structural cluster column
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

        self.assay_metadata[index] = unified_frame

    def _get_annotation(self, roi=None):
        """
        Method annotates hypothetical features using the RIO files of the National Phenome Centre

        :param roi: object (default None)
            Region of Interest (RIO) file from the National Phenome Centre.
        :return: Dataframe with structural clusters and feature annotations.
        """
        
        # Load RIO files

        for index, frame in enumerate(self.assay_metadata):

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

            # Create copies of dataframes
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

        self.assay_metadata[index] = frame_annotation
        # return frame_annotation

    def _get_isotopologue_groups(self):
        """
        Private Method.
        Searches for isotopologue signature in individual assay data frames.
        """

        for index, frame in enumerate(self.assay_metadata):
            # Create a copy of data frame
            frame = frame.copy()
            # Sort copied data frame and create new column
            frame.sort_values(by='m/z', inplace=True)
            frame.reset_index(inplace=True, drop=True)
            frame['Isotopologue group'] = [np.NaN] * len(frame)

            # Group ID for new cluster
            iso_group = 1

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
                        # If NOT NaN use current cluster ID
                        if not np.isnan(frame.loc[i, 'Isotopologue group']):
                            frame.at[j, 'Isotopologue group'] = frame.loc[i, 'Isotopologue group']

            # Update Current DataFrame
            self.assay_metadata[index] = frame

    @staticmethod
    def _mean_ppm_diff(x, y):
        """
        Calculates mean PPM difference between two numbers
        :param x: float x
        :param y: float y
        :return: mean PPM difference between x and y
        """
        diff1 = abs((x - y) / x * 1000000)
        diff2 = abs((y - x) / y * 1000000)
        return (diff1 + diff2) / 2





