# -*- coding: utf-8 -*-
#
# Authors: Lukas Kopecky <l.kopecky22@imperial.ac.uk>
#          Timothy MD Ebbels 
#          Elizabeth J Want
#
# License: BSD 3-clause

import copy as deepcopy
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from mbpls.mbpls import MBPLS
from sklearn.utils.validation import check_array, check_is_fitted
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

class MamsiPls(MBPLS):
    """
    A class that extends the MB_PLS class by extra methods convenient in Chemometrics and Metabolomics research. 
    It is based on MB-PLS package: Baum et al., (2019). Multiblock PLS: Block dependent prediction modeling for Python.
    This wrapper has some extra methods convenient in Chemometrics and Metabolomics research.

    For a full list of methods, please refer to the MB-PLS class [documentation](https://mbpls.readthedocs.io/en/latest/index.html).

    Args:
        n_components (int, optional): A number of Latent Variables (LV). Defaults to 2.
        full_svd (bool, optional): Whether to use full singular value decomposition when performing the SVD method. 
            Set to False when using very large quadratic matrices (X). Defaults to False.
        method (str, optional): The method used to derive the model attributes. Options are 'UNIPALS', 'NIPALS', 'SIMPLS', 
            and 'KERNEL'. Defaults to 'NIPALS'.
        standardize (bool, optional): Whether to standardise the data (Unit-variance scaling). Defaults to True.
        max_tol (float, optional): Maximum tolerance allowed when using the iterative NIPALS algorithm. Defaults to 1e-14.
        nipals_convergence_norm (int, optional): Order of the norm that is used to calculate the difference of 
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
            
        calc_all (bool, optional): Whether to calculate all internal attributes for the used method. Some methods do not need
            to calculate all attributes (i.e., scores, weights) to obtain the regression coefficients used for prediction.
            Setting this parameter to False will omit these calculations for efficiency and speed. Defaults to True.
        sparse_data (bool, optional): NIPALS is the only algorithm that can handle sparse data using the method of H. Martens
            and Martens (2001) (p. 381). If this parameter is set to True, the method will be forced to NIPALS and sparse data
            is allowed. Without setting this parameter to True, sparse data will not be accepted. Defaults to False.
        copy (bool, optional): Whether the deflation should be done on a copy. Not using a copy might alter the input data
            and have unforeseeable consequences. Defaults to True.
     
    Attributes:
        n_components (int): Number of Latent Variables (LV).
        Ts_ (array): (X-Side) super scores [n,k]
        T_ (list): (X-Side) block scores [i][n,k]
        W_ (list): (X-Side) block weights [n][pi,k]
        A_ (array): (X-Side) block importances/super weights [i,k]
        A_corrected_ (array): (X-Side) normalized block importances (see mbpls documentation)
        P_ (list, block): (X-Side) loadings [i][pi,k]
        R_ (array): (X-Side) x_rotations R=W(PTW)-1
        explained_var_x_ (list): (X-Side) explained variance in X per LV [k]
        explained_var_xblocks_ (array): (X-Side) explained variance in each block Xi [i,k]
        beta_ (array): (X-Side) regression vector 𝛽 [p,q]
        U_ (array): (Y-Side) scoresInitialize [n,k]
        V_ (array): (Y-Side) loadings [q,k]
        explained_var_y_ (list): (Y-Side) explained variance in Y [k]
    """

    def __init__(self, n_components=2, full_svd=False, method='NIPALS', standardize=True, max_tol=1e-14,
                 nipals_convergence_norm=2, calc_all=True, sparse_data=False, copy=True):
        
        super().__init__(n_components, full_svd, method, standardize, max_tol, nipals_convergence_norm,
                         calc_all, sparse_data, copy)
        

    def estimate_lv(self, x, y, groups=None, max_components=10, classification=True, metric='auc', 
                    method='kfold', n_splits=5, repeats=100, test_size=0.2, random_state=42, 
                    plateau_threshold=0.01, increase_threshold=0.05, get_scores=False, savefig=False, n_jobs=-1,  **kwargs):
        """A method to estimate the number of latent variables (LVs)/components in the MB-PLS model.
           The method is based on cross-validation (k-fold or Monte Carlo) and combined with an outer loop with increasing number of LVs.
           LV on which the model stabilises corresponds with the optimal number of LVs.

        Args:
            x (array or list['array']): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): A 1-dim array of reference values, either continuous or categorical variable.
            groups (array, optional): If provided, cv iterator variant with non-overlapping groups. 
                Group labels for the samples used while splitting the dataset into train/test set.
                Defaults to None.
            max_components (int, optional): Maximum number of components for whic LV estimate is calculated. 
                Defaults to 10.
            classification (bool, optional): Whether to perfrom calssification or regression. Defaults to True.
            metric (str, optional): Metric to use to estimate the number of LVs; available options: [`AUC`, `precision`, `recall`, `f1`] for 
                categorical outcome variables and ['q2'] for continuous outcome variable. 
                Defaults to 'auc'.
            method (str, optional): Corss-validation method. Available options ['kfold', 'montecarlo']. Defaults to 'kfold'.
            n_splits (int, optional): Number of splits for k-fold cross-validation. Defaults to 5.
            repeats (int, optional): Number of train-test split repeats from Monte Carlo. Defaults to 100.
            test_size (float, optional): Test size for Monte Carlo. Defaults to 0.2.
            random_state (int, optional): Generates a sequence of random splits to control MCCV. Defaults to 42.
            plateau_threshold (float, optional): Maximum increase for a sequence of LVs to be considered a plateau. 
                Must be non-negative. 
                Defaults to 0.01.
            increase_threshold (float, optional): Minimum increase to be considered a bend. Must be non-negative.. Defaults to 0.05.
            get_scores (bool, optional): Whether to retun measured mean scores. Defaults to False.
            savefig (bool, optional): Whether to save the plot as a figure. If True, argument `fname` has to be provided. Defaults to False.
            n_jobs(int, optional): Number of workers (CPU cores) for multiprocessing, -1 utilises all available cores on a system. 
                Defaults to -1.
            **kwargs: Additional keyword arguments to be passed to plt.savefig(), fname required to save .

        Raises:
            ValueError: Incorrect metric for categorical outcome. Allowed values are: 'auc', 'precision', 'recall', 'specificity', 'f1', 'accuracy'.
            ValueError: Incorrect metric for continuous outcome. Allowed values are: 'q2'.
            ValueError: Invalid method. Available options are ['kfold', 'montecarlo'].

        Returns:
            pandas.DataFrame: Measured mean scores for test and train splits for all components.
        """

        check_is_fitted(self, 'beta_')

        # Validation of data inputs
        _x = x.copy()
        if isinstance(_x, list) and not isinstance(_x[0], list):
            pass
        else:
            _x = [_x]
        _y = y.copy()
        _y = check_array(_y, ensure_2d=False)

        # Validation in parameter inputs
        if classification:
            allowed_metrics = ['auc', 'precision', 'recall', 'specificity', 'f1', 'accuracy']
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric for categorical outcome. Allowed values are: "
                                 f"{', '.join(allowed_metrics)}")
        else:
            metric = 'q2'
            allowed_metrics = ['q2']
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric continuous outcome. Allowed values are: {', '.join(allowed_metrics)}")
    
        # Scores placeholders
        training_means = pd.DataFrame()
        testing_means = pd.DataFrame()

        # Outer loop. Calculate scores for each latent variable.
        for i in range(1, max_components + 1):

            self.n_components = i
            if method == 'kfold':
                test_scores, train_scores = self.kfold_cv(x, y, groups=groups, classification=classification, return_train=True, n_splits=n_splits, n_jobs=n_jobs)
            elif method == 'montecarlo':
                test_scores, train_scores = self.montecarlo_cv(x, y, groups=groups, classification=classification, return_train=True, 
                                                                           test_size=test_size, repeats=repeats, random_state=random_state, n_jobs=n_jobs)
            else:
                raise ValueError("Invalid method. Available options are ['kfold', 'montecarlo']")

            test_scores = test_scores.dropna(axis=0)
            train_scores = train_scores.dropna(axis=0)

            if len(training_means) == 0:
                testing_means = pd.DataFrame(test_scores.mean()).T
                training_means = pd.DataFrame(train_scores.mean()).T
            else:
                testing_means = pd.concat([testing_means, pd.DataFrame(test_scores.mean()).T])
                training_means = pd.concat([training_means, pd.DataFrame(train_scores.mean()).T])

        testing_means.reset_index(drop=True, inplace=True)
        training_means.reset_index(drop=True, inplace=True)

        # concatenate training and testing scores into one dataframe, add prefix training_ to training scores
        perf_scores = pd.concat([training_means.add_prefix('training_'), testing_means], axis=1)
        perf_scores.insert(0, 'Number of Components', range(1, max_components + 1))

        #  for regression, delete RMSE from the dataframe and rename training_q2 to r2
        if not classification:
            perf_scores.drop('training_rmse', axis=1, inplace=True)
            perf_scores.drop('rmse', axis=1, inplace=True)
            perf_scores.rename(columns={'training_q2': 'r2'}, inplace=True)
            
        # Select desired metric
        if metric == 'q2':
            _data = testing_means['q2']
        if metric == 'auc':
            _data = testing_means['roc_auc']
        if metric == 'precision':
            _data = testing_means['precision']
        if metric == 'recall':
            _data = testing_means['recall']
        if metric == 'f1':
            _data = testing_means['f1']
        if metric == 'accuracy':
            _data = testing_means['accuracy']
        if metric == 'specificity':
            _data = testing_means['specificity']

        # Estimate number of LVs
        try:
            bend = np.min(np.where(np.diff(_data) / _data[0] < increase_threshold)[0]) + 1
        except ValueError:
            # Handle the case where np.min() fails due to an empty array
            bend = 1
        plateau_range_start, plateau_range_end = self._find_plateau(_data, range_threshold=plateau_threshold)

        # Percentage for printed statements below
        increase = increase_threshold * 100

        # Plot the results
        perf_scores.plot.line(x='Number of Components', marker='.', figsize=(8, 6), grid=False)
        plt.xlim(0, max_components + 1)
        plt.xticks(np.arange(1, max_components + 1, 1.0))
        try:
            plt.axvline(plateau_range_start, linestyle='--', color='r', label='Plateau edge')
            plt.axvline(bend, linestyle='dotted', color='b', label='Bend')
            print(metric + " reaches bent (increase of less than {0}".format(increase) +
                  " % of previous value or decrease) at component {0}".format(bend))
            print(metric + " reaches plateau at component {0}".format(plateau_range_start))
            self.n_components = plateau_range_start
            self.fit(x, y)
            print("Model re-fitted with n_components =", self.n_components)
        except TypeError:
            print(metric + " reaches bend (increase of less than {0}".format(increase) +
                  " % of previous value or decrease) at component {0}".format(bend))
            plt.axvline(bend, linestyle='--', color='b', label='Bend')
            print('No plateau detected, consider exploring more latent variables.')
            self.n_components = bend
            self.fit(x, y)
            print("Model re-fitted with n_components =", self.n_components)
        plt.ylabel('Score')
        plt.xlabel('Number of Latent Variables')
        if method == 'kfold':
            title = 'Latent Variable Estimation' + ' (k-Fold) '
        else:
            title = 'Latent Variable Estimation' + ' (Monte Carlo) ' 
        plt.title(title)
        plt.legend()

        if savefig:
            plt.savefig(**kwargs)

        if get_scores:
            return perf_scores    


    def evaluate_class_model(self, x, y):
        """
        Evaluate classification MB-PLS model using a **testing** dataset.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): 1-dim or 2-dim array of reference values - categorical variable.

        Returns:
            array: Predicted y variable based on training set predictors.
        """

        # Check if PLS model is fitted
        check_is_fitted(self, 'beta_')

        # Validate inputs
        _x = x.copy()
        if isinstance(_x, list) and not isinstance(_x[0], list):
            pass
        else:
            _x = [x]
        _y = y.copy()
        _y = check_array(_y, ensure_2d=False)

        # Predict test data
        y_predicted = self.predict(_x)

        # Evaluation metrics
        cm2 = confusion_matrix(y, np.where(y_predicted > 0.5, 1, 0))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
        disp2.plot()
        print('Scores for model with n_components =', self.n_components)
        print('\nAccuracy', round(accuracy_score(_y, np.where(y_predicted > 0.5, 1, 0)), 3))
        print('Precision', round(precision_score(_y, np.where(y_predicted > 0.5, 1, 0)), 3))
        print('Recall', round(recall_score(_y, np.where(y_predicted > 0.5, 1, 0)), 3))
        tn, fp, fn, tp = confusion_matrix(_y, np.where(y_predicted > 0.5, 1, 0)).ravel()
        print('Specificity', round(tn/(tn+fp), 3))
        print('F1 Score', round(f1_score(_y, np.where(y_predicted > 0.5, 1, 0)), 3))
        print('AUC', round(roc_auc_score(_y, y_predicted), 3))
        return y_predicted
    
    def evaluate_regression_model(self, x, y):
        """
        Evaluate regression MB-PLS model using a **testing** dataset.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): 1-dim or 2-dim array of reference values - continuous variable.

        Returns:
            array: Predicted y variable based on training set predictors.
        """

         # Check if PLS model is fitted
        check_is_fitted(self, 'beta_')

        # Validate inputs
        _x = x.copy()
        if isinstance(_x, list) and not isinstance(_x[0], list):
            pass
        else:
            _x = [x]
        _y = y.copy()
        _y = check_array(_y, ensure_2d=False)

        # Predict test data
        y_predicted = self.predict(_x)

        # Evaluation metrics
        rmse = root_mean_squared_error(_y, y_predicted)
        print(f'Root Mean Squared Error: {rmse}')
        q2 = r2_score(_y, y_predicted)
        print(f'Q-squared: {q2}')

        # Plotting Regresseion model evaluation
        plt.figure(dpi=600, figsize=(10, 6))
        plt.scatter(y_predicted, _y)
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.title('Regression Model Evaluation')

        return y_predicted
    
    def kfold_cv(self, x, y, groups=None, classification=True, return_train=False, n_splits=5, n_jobs=-1):
        """
        Perform k-fold cross-validation for MB-PLS model.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): 1-dim or 2-dim array of reference values, either continuous or categorical variable.
            groups (array, optional): Group labels for the samples used while splitting the dataset into train/test set.
                If provided, group k-fold is performed. 
                Defaults to None.
            classification (bool, optional): Whether the outcome is a categorical variable. Defaults to True.
            return_train (bool, optional): Whether to return evaluation metrics for training set. Defaults to False.
            n_splits (int, optional): Number of splits for k-fold cross-validation. Defaults to 5.
            n_jobs (int, optional): Number of workers (CPU cores) for multiprocessing, -1 utilises all available cores on a system. 
                Defaults to -1.

        Returns:
            pandas.DataFrame: Evaluation metrics for each k-fold split.
                if return_train is True, returns evaluation metrics for training set as well.
        """

        check_is_fitted(self, 'beta_')

        # Validate inputs
        x_cp = x.copy()
        if isinstance(x_cp, list) and not isinstance(x_cp[0], list):
            pass
        else:
            x_cp = [x_cp]
        y_cp = y.copy()
        y_cp = check_array(y_cp, ensure_2d=False)

        # if groups are provided, group k-fold is performed, otherwise sk-learn k-fold is used
        if groups is None:
            kf = KFold(n_splits=n_splits)
        else:
            kf = GroupKFold(n_splits=n_splits)

        scores = pd.DataFrame()    

        def _kfold (_x, _y, _j, _train_indices, _test_indices, _classification, _return_train):
        
        # (j, (train_indices, test_indices) in enumerate(kf.split(_y, groups=groups))):
            # Unwrap data perform test-train split
            x_train = [None] * len(_x)
            x_test = [None] * len(_x)
            train_test_data = [None] * len(_x)
            y_train = _y[_train_indices]
            y_test = _y[_test_indices]

            for k in range(len(_x)):
                x_train[k] = pd.DataFrame(_x[k]).iloc[_train_indices]  # filter training data by index and save in a new list
                x_test[k] = pd.DataFrame(_x[k]).iloc[_test_indices]  # filter testing data by index and save in a new list
                train_test_data[k] = pd.DataFrame(_x[k]).iloc[_train_indices]

            # Fit model and predict
            x_train_copy = deepcopy.deepcopy(x_train)
            self.fit_transform(x_train_copy, y_train)# for each n_components fit new model
    
            # Predict outcome based on training folds
            y_predicted = self.predict(x_test)
            predictions = [y_predicted]
            truths = [y_test]

            # add training scores
            if _return_train:
                y_predicted_train = self.predict(x_train)
                predictions.append(y_predicted_train)
                truths.append(y_train)

            # Classification model evaluation
            if _classification:
                predictions_cl = [np.where(y_predicted > 0.5, 1, 0)]
                if _return_train:
                    predictions_cl.append(np.where(y_predicted_train > 0.5, 1, 0))

                # Calculate evaluation metrics for testing and training sets
                for prediction_cl, prediction, truth, j in zip(predictions_cl, predictions, truths, [0,1]):
                    # Evaluation metrics
                    try:
                        accuracy = accuracy_score(truth, prediction_cl)
                    except ValueError:
                        accuracy = np.nan
                    try:
                        precision = precision_score(truth, prediction_cl)
                    except ValueError:
                        precision = np.nan
                    try:
                        recall = recall_score(truth, prediction_cl, zero_division=np.nan)
                    except ValueError:
                        recall = np.nan
                    try:
                        f1 = f1_score(truth, prediction_cl)
                    except ValueError:
                        f1 = np.nan
                    try:
                        tn, fp, _, _ = confusion_matrix(truth, prediction_cl, labels=[0, 1]).ravel()
                        specificity_score = tn/(tn+fp) if (tn + fp) != 0 else np.nan
                    except ValueError:
                        specificity_score = np.nan
                    try:
                        roc_auc = roc_auc_score(truth, prediction)
                    except ValueError:
                        roc_auc = np.nan

                    row = np.array([[precision, recall, specificity_score, f1, roc_auc, accuracy]])

                    if j == 0:
                        # save MCCV scores
                        test_score_row = row.copy()
                    else:
                        train_score_row = row.copy()

            # Regression model evaluation    
            else:

                for prediction, truth, j in zip(predictions, truths, [0,1]):
                    # Evaluation metrics
                    rmse = root_mean_squared_error(truth, prediction)
                    q2 = r2_score(truth, prediction)

                    row = np.array([[rmse, q2]])


                    if j == 0:
                        # save MCCV scores
                        test_score_row = row.copy()
                    else:
                        train_score_row = row.copy()

            _scores = test_score_row
            if _return_train:
                _train_scores = train_score_row

            if _return_train:
                return _scores, _train_scores 

            else:
                return _scores, None

         # Run parellelised funciton 
        results = Parallel(n_jobs=n_jobs)(delayed(_kfold)
        (_x = x_cp, _y = y_cp, _j=i, _train_indices=train_indices, _test_indices=test_indices, _classification=classification, _return_train=return_train) 
        for i, (train_indices, test_indices) in enumerate(kf.split(y_cp, groups=groups)))
        
        # Get scores unwrapped 
        scores, train_scores = zip(*results)
        scores = np.stack(scores, axis=1)

        # Add lables to the the outcome labels
        if classification:
            scores = pd.DataFrame(scores[0], columns=['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'accuracy' ]) 
        else:
            scores = pd.DataFrame(scores[0], columns=['rmse', 'q2'])

        if return_train:
            train_scores = np.stack(train_scores, axis=1)
            if classification:
                train_scores = pd.DataFrame(train_scores[0], columns=['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'accuracy' ]) 
            else:
                train_scores = pd.DataFrame(train_scores[0], columns=['rmse', 'q2'])

        if return_train:
            return scores, train_scores
        else:    
            return scores


    def montecarlo_cv(self, x, y, groups=None, classification=True, return_train=False, test_size=0.2, repeats=10, random_state=42, n_jobs=-1):
        """
        Evaluate MB-PLS model using Monte Carlo Cross-Validation (MCCV).

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are featuress.
            y (array): 1-dim or 2-dim array of reference values - categorical variable.
            groups (array, optional): Group labels for the samples used while splitting the dataset into train/test set.
                If provided, group-train-test split will be used instead of train-test split for random splits. 
                Defaults to None.
            classification (bool, optional): Whether the outcome is a categorical variable. Defaults to True.
            return_train (bool, optional): Whether to return evaluation metrics for training set. Defaults to False.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            repeats (int, optional): Number of MCCV repeats. Defaults to 10.
            random_state (int, optional): Generates a sequence of random splits to control MCCV. Defaults to 42.
            n_jobs (int, optional): Number of workers (CPU cores) for multiprocessing, -1 utilises all available cores on a system. 
                Defaults to -1.

        Returns:
            pandas.DataFrame: Evaluation metrics for each MCCV repeat.
                if return_train is True, returns evaluation metrics for training set as well.
        """

        # Check if PLS model is fitted
        check_is_fitted(self, 'beta_')

        # Validate inputs
        x_cp = x.copy()
        if isinstance(x_cp, list) and not isinstance(x_cp[0], list):
            pass
        else:
            x_cp = [x_cp]
        y_cp = y.copy()
        y_cp = check_array(y_cp, ensure_2d=False)

        # Generate random sequence of seeds for MCCV
        rng = np.random.RandomState(random_state)
        
        # Generate n random numbers
        random_numbers = rng.randint(1, 2_147_483_647, size=repeats)

        # Placeholder for MCCV scores
        scores = pd.DataFrame()

        def _mccv(_x, _y, i, _groups, _classification, _return_train, _test_size, _random_numbers):


            if _groups is None:
                train, test, y_train, y_test = train_test_split(pd.DataFrame(_x[0]), _y, test_size=_test_size, random_state=_random_numbers[i])
                if isinstance(_x[0], pd.DataFrame):
                    x_train = [df.loc[train.index] for df in _x]
                    x_test = [df.loc[test.index] for df in _x]
                else:
                    x_train = [df[train.index] for df in _x]
                    x_test = [df[train.index] for df in _x]
            else:
                x_train, x_test, y_train, y_test = self.group_train_test_split(_x, _y, groups=_groups, test_size=_test_size, random_state=_random_numbers[i])

            # Fit model and predict
            x_train_copy = deepcopy.deepcopy(x_train)
            self.fit_transform(x_train_copy, y_train)

            # Predict outcome based on training folds
            y_predicted = self.predict(x_test)
            predictions = [y_predicted]
            truths = [y_test]

            # add training scores
            if _return_train:
                y_predicted_train = self.predict(x_train)
                predictions.append(y_predicted_train)
                truths.append(y_train)

            # Classification model evaluation
            if _classification:
                predictions_cl = [np.where(y_predicted > 0.5, 1, 0)]
                if _return_train:
                    predictions_cl.append(np.where(y_predicted_train > 0.5, 1, 0))

                # Calculate evaluation metrics for testing and training sets
                for prediction_cl, prediction, truth, j in zip(predictions_cl, predictions, truths, [0,1]):
                    # Evaluation metrics
                    try:
                        accuracy = accuracy_score(truth, prediction_cl)
                    except ValueError:
                        accuracy = np.nan
                    try:
                        precision = precision_score(truth, prediction_cl)
                    except ValueError:
                        precision = np.nan
                    try:
                        recall = recall_score(truth, prediction_cl, zero_division=np.nan)
                    except ValueError:
                        recall = np.nan
                    try:
                        f1 = f1_score(truth, prediction_cl)
                    except ValueError:
                        f1 = np.nan
                    try:
                        tn, fp, _, _ = confusion_matrix(truth, prediction_cl, labels=[0, 1]).ravel()
                        specificity_score = tn/(tn+fp) if (tn + fp) != 0 else np.nan
                    except ValueError:
                        specificity_score = np.nan
                    try:
                        roc_auc = roc_auc_score(truth, prediction)
                    except ValueError:
                        roc_auc = np.nan

                    row = np.array([[precision, recall, specificity_score, f1, roc_auc, accuracy]])

                    if j == 0:
                        # save MCCV scores
                        test_score_row = row.copy()
                    else:
                        train_score_row = row.copy()

            # Regression model evaluation    
            else:

                for prediction, truth, j in zip(predictions, truths, [0,1]):
                    # Evaluation metrics
                    rmse = root_mean_squared_error(truth, prediction)
                    q2 = r2_score(truth, prediction)

                    row = np.array([[rmse, q2]])

                    if j == 0:
                        # save MCCV scores
                        test_score_row = row.copy()
                    else:
                        train_score_row = row.copy()
            
            _scores = test_score_row
            if _return_train:
                _train_scores = train_score_row

            if _return_train:
                return _scores, _train_scores 

            else:
                return _scores, None

        # Run parellelised funciton 
        results = Parallel(n_jobs=n_jobs)(delayed(_mccv)(_x = x_cp, _y=y_cp, i=j, _groups=groups, _classification=classification, _return_train=return_train, _test_size=test_size, _random_numbers=random_numbers) for j in range(repeats))
        
        # Get scores unwrapped 
        scores, train_scores = zip(*results)
        scores = np.stack(scores, axis=1)

        # Add lables to the the outcome labels
        if classification:
            scores = pd.DataFrame(scores[0], columns=['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'accuracy' ]) 
        else:
            scores = pd.DataFrame(scores[0], columns=['rmse', 'q2'])

        if return_train:
            train_scores = np.stack(train_scores, axis=1)
            if classification:
                train_scores = pd.DataFrame(train_scores[0], columns=['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'accuracy' ]) 
            else:
                train_scores = pd.DataFrame(train_scores[0], columns=['rmse', 'q2'])

        # retun data
        if return_train:
            return scores, train_scores
        else:    
            return scores
            

    def mb_vip(self, plot=True, get_scores=False, savefig=False, **kwargs):
        """
        Multi-block Variable Importance in Projection (MB-VIP) for multiblock PLS model.

        Adaptation of C. Wieder et al., (2024). PathIntegrate, doi: 10.1371/journal.pcbi.1011814.

        Args:
            plot (bool, optional): Whether to plot MB-VIP scores. Defaults to True.
            get_scores (bool, optional): Whether to return MB-VIP scores. Defaults to False.
            savefig (bool, optional): Whether to save the plot as a figure. If True, argument `fname` has to be provided. 
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to plt.savefig(), fname required to save .            

        Returns:
            array: MB-VIP scores.
        """

        # Check is model is fitted
        check_is_fitted(self, 'beta_')

        # stack the weights from all blocks
        weights = np.vstack(self.W_)
        # calculate product of sum of squares of super scores and y loadings
        sum_squares = np.sum(self.Ts_ ** 2, axis=0) * np.sum(self.V_ ** 2, axis=0)
        # p = number of variables - stack the loadings from all blocks
        p = np.vstack(self.P_).shape[0]
        # VIP is a weighted sum of squares of PLS weights
        vip_scores = np.sqrt(p * np.sum(sum_squares * (weights ** 2), axis=1) / np.sum(sum_squares))

        # Plot VIP scores
        if plot:
            plt.plot(vip_scores, color='limegreen', linewidth=0.8)
            plt.title('Multi-block variable importance in projection')
            plt.ylabel('MB-VIP score')
            plt.xlabel('Feature index')

            if savefig:
                plt.savefig(**kwargs)

        # Return all MB-VIP scores
        if get_scores:
            return vip_scores
        
            
    def block_importance(self, block_labels=None, normalised=True, plot=True, get_scores=False, savefig=False, **kwargs):
        '''
        Calculate the block importance for each block in the multiblock PLS model and plot the results.
        
        Args:
            block_labels (list, optional): List of block names. If block names are not provided or they do not match the number 
                of blocks in the model, the plot will display labels as 'Block 1', 'Block 2', ... 'Block n'. Defaults to None.
            normalised (bool, optional): Whether to use normalised block importance. For more information see model attribute 
                ['A_Corrected_'](). Defaults to True.
            plot (bool, optional): Whether to render plot block importance. Defaults to True.
            get_scores (bool, optional): Whether to return block importance scores. Defaults to False.
            savefig (bool, optional): Whether to save the plot as a figure. If True, argument `fname` has to be provided. 
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to plt.savefig(), fname required to save.
        
        Returns:
            array: Block importance scores.
        '''
        
        # Check if PLS model is fitted
        check_is_fitted(self, 'beta_')

        # get the block importances
        if normalised:
            block_importance = self.A_corrected_
        else:
            block_importance = self.A_
        block_importance_t = block_importance.T

        if plot:
            # Number of groups and number of bars in each group
            num_groups = len(block_importance)
            num_bars = len(block_importance[0])

            # Set up the figure and axis
            fig, ax = plt.subplots()

            # Set the width of the bars
            bar_width = 0.15  # Decreased bar width

            # Calculate the total width for each group of bars
            total_bar_width = bar_width * num_bars

            # Increase spacing between groups
            group_spacing = 0.5

            # Set the positions for the bars with offset to prevent overlap
            bar_positions = np.arange(num_groups) * (total_bar_width + group_spacing)

            # Plot each group of bars
            for i in range(num_bars):
                ax.bar(bar_positions + i * bar_width, block_importance_t[i], width=bar_width, label=f'LV {i + 1}')

            # Set labels and title
            ax.set_xlabel('Blocks')
            ax.set_ylabel('Explained variance')
            ax.set_title('Block Importance')
            ax.set_xticks(bar_positions + (total_bar_width / 2))

            # Set block names as x-tick labels
            if block_labels is None:
                # call blocks 'Block 1', 'Block 2' etc. if no names are provided
                block_labels = [f'Block {i + 1}' for i in range(num_groups)]
                ax.set_xticklabels(block_labels)

            elif len(block_labels) != num_groups:
                # call blocks 'Block 1', 'Block 2' etc. if length of block names does not match the number of blocks
                block_labels = [f'Block {i + 1}' for i in range(num_groups)]
                ax.set_xticklabels(block_labels)
                raise ValueError('Number of block names must match the number of blocks')
            
            else:
                # set block names as x-tick labels
                ax.set_xticklabels(block_labels)

            # Add legend
            ax.legend()

            # Show the plot
            plt.tight_layout()  # Adjust layout to prevent clipping of labels
        

            if savefig:
                plt.savefig(**kwargs)

            plt.show()

        if get_scores:
            return block_importance
        

    def mb_vip_permtest(self, x, y, n_permutations=1000, return_scores=False, n_jobs=-1):
        """
        Calculate empirical p-values for each feature by permuting the Y outcome variable `n_permutations` times and
        refitting the model. The p-values for each feature are calculated by counting the number of trials with
        MB-VIP greater than or equal to the observed test statistic, and dividing this by `n_permutations`.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): 1-dim or 2-dim array of reference values, either continuous or categorical variable.
            n_permutations (int, optional): Number of permutation tests. Defaults to 1000.
            return_scores (bool, optional): Whether to return MB-VIP scores for each permuted null model. Defaults to False.
            n_jobs (int, optional): Number of workers (CPU cores) for multiprocessing, -1 utilises all available cores on a system. 
                Defaults to -1.

        Returns:
            array: Returns an array of p-values for each feature. If `return_scores` is True, then a matrix of MB-VIP scores
            for each permuted null model is returned as well.
        """


        # Check is model is fitted
        check_is_fitted(self, 'beta_')

        # Validation of data inputs
        _x = deepcopy.deepcopy(x)  # deepcopy to prevent data leakage
        if isinstance(_x, list) and not isinstance(_x[0], list):
            pass
        else:
            _x = [_x]
        _y = y.copy()
        _y = check_array(_y, ensure_2d=False)

        # MB-VIP of observed model
        _vip = self.mb_vip(plot=False, get_scores=True)
        vip_obs = _vip[:, np.newaxis]

        def _fit_permute(x, y):
            y_perm = np.random.permutation(y)
            self.fit(x, y_perm)
            return self.mb_vip(plot=False, get_scores=True)

        _vip_null = Parallel(n_jobs=n_jobs)(delayed(_fit_permute)(_x, _y) for _ in range(n_permutations))
        vip_null = np.stack(_vip_null, axis=1)

        # Calculate empirical p-values
        vip_greater = np.sum(vip_null >= vip_obs, axis=1)
        p_vals = vip_greater/n_permutations

        # Return p-vales and MB-PLS scores for null models
        if return_scores:
            return p_vals, vip_null
        else:
            return p_vals

    @staticmethod
    def calculate_ci(data, ci_level=0.90, dropna=True):
        """
        Static Method
        
        Calculates mean, margin of error, and confidence interval for each column.

        Args:
            data (pd.DataFrame): The input DataFrame.
            ci_level (float, optional): The confidence level (e.g., 0.90, 0.95). Defaults to 0.90.
            dropna (bool, optional): Whether to drop rows containing NaNs. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated statistics for each column.
                        If dropna = False, and a column has less than 2 valid values after
                        dropping NaNs specific to that column, all the result values for that
                        column will be np.nan.
        """

        if dropna:
            _data = data.dropna()  # Drop rows with NaNs for consistent comparison
        else:
            _data = data  # Use the data as-is (handle NaNs later if needed)

        results = {}
        for col in _data.columns:
            col_data = _data[col]
            if not dropna: # if user did not specify dropna, handle nans in individual columns
                col_data = col_data.dropna() # drop for each column separately
            
            if len(col_data) < 2: # handle cases where after dropping nans for a column, there is only one element. 
                print(f"Warning: Column {col} has less than 2 valid values after NaN removal. Cannot calculate CI.")
                results[col] = {"mean": np.nan, "margin_of_error": np.nan, "CI_lower": np.nan, "CI_upper": np.nan}
                continue

            mean = np.mean(col_data)
            sem = stats.sem(col_data)
            ci = stats.t.interval(ci_level, len(col_data) - 1, loc=mean, scale=sem)  # len(col_data) - 1: Degrees of freedom
            error = (ci[1] - ci[0]) / 2
            results[col] = {"mean": mean, "margin_of_error": error, "CI_lower": ci[0], "CI_upper": ci[1]}

        return pd.DataFrame(results)

    @staticmethod
    def group_train_test_split(x, y, gropus=None, test_size=0.2, random_state=42):
        """
        Static Method

        Split the data into train and test sets based on the groups. The groups are split into train and test sets
        based on the `test_size` parameter. The function returns the train and test sets for the predictors and the
        response variable.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are samples, columns are features.
            y (array): 1-dim or 2-dim array of reference values, either continuous or categorical variable.
            groups (array, optional): Group labels for the samples used while splitting the dataset into train/test set. Defaults to None.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

        Returns:
            tuple: x_train, x_test, y_train, y_test
        """
    
        # Ensure x is a list of data frames
        if not isinstance(x, list):
            x = [x]

        if groups is not None:
            groups = check_array(groups, ensure_2d=False, dtype=None)

        # Split the groups into train and test
        unique_groups = pd.unique(groups)
        train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)

        # Create masks for train and test
        train_mask = np.isin(groups, train_groups)
        test_mask = np.isin(groups, test_groups)

        # Split the data frames in x
        x_train = [df.loc[train_mask] for df in x]
        x_test = [df.loc[test_mask] for df in x]

        # Split y
        y_train = y[train_mask]
        y_test = y[test_mask]

        return x_train, x_test, y_train, y_test
    

    @staticmethod
    def _find_plateau(scores, range_threshold=0.01, consecutive_elements=3):
        """
        Private Method, Static Method
        
        Function to assist in finding a plateau in a sequence of LVs.

        Args:
            scores (list[float]): List of scores.
            range_threshold (float, optional): Maximum increase for a sequence of LVs to be considered a plateau. Defaults to 0.01.
            consecutive_elements (int, optional): Number of elements that need to be in a plateau. Defaults to 3.

        Returns:
            tuple: Beginning and end indices of the plateau.
        """

        n = len(scores)
        for i in range(1, n - consecutive_elements + 1):
            plateau = True
            for j in range(consecutive_elements - 1):
                diff = abs(scores[i + j] - scores[i + j - 1])
                if diff > range_threshold:
                    plateau = False
                    break
            if plateau:
                return i, i + consecutive_elements - 1

        # If no plateau is found, return a tuple of None
        return None, None
