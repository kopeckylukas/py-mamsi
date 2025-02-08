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
        

    def estimate_lv(self, x, y, groups=None, max_components=10, method='kfold', n_splits=5, y_continuous=False, metric='auc',
                    plateau_threshold=0.01, increase_threshold=0.05, get_scores=False, savefig=False, random_state=42, **kwargs):
        """
        A method to estimate the number of latent variables (LVs)/components in the MB-PLS model.
        The method is based k-fold cross-validation and combined with an outer loop with increasing number of LVs.
        LV on which the model stabilises corresponds with the optimal number of LVs.

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): A 1-dim or 2-dim array of reference values, either continuous or categorical variable.
            groups (array, optional): If provided, K-fold iterator variant with non-overlapping groups. 
                Group labels for the samples used while splitting the dataset into train/test set.
                Defaults to None. 
            max_components (int, optional): Number of components/LVs. Defaults to 10.
            n_splits (int, optional): Number of splits for k-fold cross-validation. Defaults to 5.
            y_continuous (bool, optional): Whether the outcome is a continuous variable. Defaults to False.
            metric (str, optional): Metric to use to estimate the number of LVs; available options: [`AUC`, `precision`, `recall`, `f1`] for 
                categorical outcome variables and ['q2'] for continuous outcome variable. 
                Defaults to 'AUC'.
            plateau_threshold (float, optional): Maximum increase for a sequence of LVs to be considered a plateau. Must be non-negative. Defaults to 0.01.
            increase_threshold (float, optional): Minimum increase to be considered a bend. Must be non-negative. Defaults to 0.05.
            get_scores (bool, optional): Whether to return measured scores as a Pandas DataFrame. Defaults to False.
            savefig (bool, optional): Whether to save the plot as a figure. If True, argument `fname` has to be provided. 
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to plt.savefig(), fname required to save .
        Returns:
            pandas.DataFrame: Measured mean scores for test and train splits for all components returned as a Pandas DataFrame.
        """

        check_is_fitted(self, 'beta_')

        # Validation of data inputs
        data = x.copy()
        # Check if the data is a list of a pandas.DataFrame
        if isinstance(data, list) and not isinstance(data[0], list):
            pass
        else:
            data = [data]

        _y = check_array(y, ensure_2d=False)
        response_y = _y.copy()

        # Validation in parameter inputs
        if y_continuous:
            metric = 'q2'
            allowed_metrics = ['q2']
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric continuous outcome. Allowed values are: {', '.join(allowed_metrics)}")
        else:
            allowed_metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy']
            if metric not in allowed_metrics:
                raise ValueError(f"Invalid metric for categorical outcome. Allowed values are: "
                                 f"{', '.join(allowed_metrics)}")

        # Scores placeholders
        r2 = []
        q2 = []
        r2_auc = []
        r2_precision = []
        r2_recall = []
        r2_f1 = []
        r2_accuracy = []
        q2_auc = []
        q2_precision = []
        q2_recall = []
        q2_f1 = []
        q2_accuracy = []

        #If `groups` are provided, group k-fold is performed
        if groups is None:
            kf = KFold(n_splits=n_splits)
        else:
            kf = GroupKFold(n_splits=n_splits)

        # Estimation of scores for different number of latent variables / components.
        for i in range(1, max_components + 1):

            # Set LV scores placeholder lists
            lv_r2 = []
            lv_q2 = []
            lv_r2_precision = []
            lv_r2_recall = []
            lv_r2_auc = []
            lv_r2_f1 = []
            lv_r2_accuracy = []
            lv_q2_precision = []
            lv_q2_recall = []
            lv_q2_auc = []
            lv_q2_f1 = []
            lv_q2_accuracy = []
            
            if method == 'kfold':
                for j, (train_indices, test_indices) in enumerate(kf.split(response_y, groups=groups)):

                    # Unwrap data perform test-train split
                    train_data = [None] * len(data)
                    test_data = [None] * len(data)
                    train_test_data = [None] * len(data)
                    for k in range(len(data)):
                        train_data[k] = data[k].iloc[train_indices]  # filter training data by index and save in a new list
                        test_data[k] = data[k].iloc[test_indices]  # filter testing data by index and save in a new list
                        train_test_data[k] = data[k].iloc[train_indices]

                    # for each n_components fit new model
                    self.n_components = i
                    self.fit_transform(train_data, response_y[train_indices])

                    # Predict outcome based on training folds
                    y_predicted_train = self.predict(train_test_data)
                    # Calculate predictive performance of training folds
                    if y_continuous:
                        lv_r2.append(r2_score(response_y[train_indices], y_predicted_train))
                    else:
                        lv_r2_auc.append(roc_auc_score(response_y[train_indices], y_predicted_train))
                        lv_r2_precision.append(precision_score(response_y[train_indices],
                                                            np.where(y_predicted_train > 0.5, 1, 0)))
                        lv_r2_recall.append(
                            recall_score(response_y[train_indices], np.where(y_predicted_train > 0.5, 1, 0)))
                        lv_r2_f1.append(f1_score(response_y[train_indices], np.where(y_predicted_train > 0.5, 1, 0)))
                        lv_r2_accuracy.append(
                            accuracy_score(response_y[train_indices], np.where(y_predicted_train > 0.5, 1, 0)))

                    # Predict outcome based on testing folds
                    y_predicted_test = self.predict(test_data)
                    # Calculate predictive performance of testing folds
                    if y_continuous:
                        lv_q2.append(r2_score(response_y[test_indices], y_predicted_test))
                    else:
                        lv_q2_auc.append(roc_auc_score(response_y[test_indices], y_predicted_test))
                        lv_q2_precision.append(precision_score(response_y[test_indices],
                                                            np.where(y_predicted_test > 0.5, 1, 0)))
                        lv_q2_recall.append(recall_score(response_y[test_indices], np.where(y_predicted_test > 0.5, 1, 0)))
                        lv_q2_f1.append(f1_score(response_y[test_indices], np.where(y_predicted_test > 0.5, 1, 0)))
                        lv_q2_accuracy.append(
                            accuracy_score(response_y[test_indices], np.where(y_predicted_test > 0.5, 1, 0)))
            
            # Calculate mean scores of predictive performance for training and testing folds across for each LV
            if y_continuous:
                r2.append(statistics.mean(lv_r2))
                q2.append(statistics.mean(lv_q2))
            else:
                r2_auc.append(statistics.mean(lv_r2_auc))
                q2_auc.append(statistics.mean(lv_q2_auc))
                r2_precision.append(statistics.mean(lv_r2_precision))
                r2_recall.append(statistics.mean(lv_r2_recall))
                r2_f1.append(statistics.mean(lv_r2_f1))
                r2_accuracy.append(statistics.mean(lv_r2_accuracy))
                q2_precision.append(statistics.mean(lv_q2_precision))
                q2_recall.append(statistics.mean(lv_q2_recall))
                q2_f1.append(statistics.mean(lv_q2_f1))
                q2_accuracy.append(statistics.mean(lv_q2_accuracy))

        if y_continuous:
            perf_scores = pd.DataFrame([range(1, max_components + 1), r2, q2],
                                       index=['Number of Components', 'r2', 'q2']).T
        else:
            perf_scores = pd.DataFrame([range(1, max_components + 1), r2_auc, r2_precision, r2_recall, r2_f1,
                                        r2_accuracy, q2_auc, q2_precision, q2_recall, q2_f1, q2_accuracy],
                                       index=['Number of Components', 'Training AUC', 'Training Precision',
                                              'Training Recall', 'Training F1 Score', 'Training Accuracy',
                                              'Testing AUC', 'Testing Precision', 'Testing Recall',
                                              'Testing F1 Score', 'Testing Accuracy']).T

        # Select desired metric
        if metric == 'q2':
            data = q2
        if metric == 'auc':
            data = q2_auc
        if metric == 'precision':
            data = q2_precision
        if metric == 'recall':
            data = q2_recall
        if metric == 'f1':
            data = q2_f1
        if metric == 'accuracy':
            data = q2_accuracy

        # Estimate number of LVs
        try:
            bend = np.min(np.where(np.diff(data) / data[0] < increase_threshold)[0]) + 1
        except ValueError:
            # Handle the case where np.min() fails due to an empty array
            bend = 1
        plateau_range_start, plateau_range_end = self._find_plateau(data, range_threshold=plateau_threshold)

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
        plt.title('Latent Variable Estimation')
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
    
    def evaluate_class_model_mccv(self, x, y, classification=True, groups=None, return_train=False, test_size=0.2, repeats=10 , random_state=42):
        """
        Evaluate MB-PLS model using Monte Carlo Cross-Validation (MCCV).

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are featuress.
            y (array): 1-dim or 2-dim array of reference values - categorical variable.
            classification (bool, optional): Whether the outcome is a categorical variable. Defaults to True.
            groups (array, optional): Group labels for the samples used while splitting the dataset into train/test set.
                If provided, group-train-test split will be used instead of train-test split for random splits. 
                Defaults to None.
            return_train (bool, optional): Whether to return evaluation metrics for training set. Defaults to False.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            repeats (int, optional): Number of MCCV repeats. Defaults to 10.
            random_state (int, optional): Generates a sequence of random splits to control MCCV. Defaults to 42.

        Returns:
            pandas.DataFrame: Evaluation metrics for each MCCV repeat.
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

        # Generate random sequence of seeds for MCCV
        rng = np.random.RandomState(random_state)
        
        # Generate n random numbers
        random_numbers = rng.randint(1, 4294967296, size=repeats)

        # Placeholder for MCCV scores
        scores = pd.DataFrame()

        # if groups are provided, group k-fold is performed otherwise sk-learn k-fold
        for i in range(repeats):
            if groups is None:
                train, test, y_train, y_test = train_test_split(_x[0], _y, test_size=test_size, random_state=random_numbers[i])
                x_train = [df.loc[train.index] for df in _x]
                x_test = [df.loc[test.index] for df in _x]
            else:
                x_train, x_test, y_train, y_test = self.group_train_test_split(_x, _y, groups=groups, test_size=test_size, random_state=random_numbers[i])

            # Fit model and predict
            x_train_copy = deepcopy.deepcopy(x_train)
            self.fit_transform(x_train_copy, y_train)
            y_predicted = self.predict(x_test)
            predictions = [y_predicted]
            truths = [y_test]

            # add training scores
            if return_train:
                y_predicted_train = self.predict(x_train)
                predictions.append(y_predicted_train)
                truths.append(y_train)

            # Classification model evaluation
            if classification:
                predictions_cl = [np.where(y_predicted > 0.5, 1, 0)]
                if return_train:
                    predictions_cl.append(np.where(y_predicted_train > 0.5, 1, 0))

                    print(y_train)
                    print(predictions_cl[1])

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
                        recall = recall_score(truth, prediction_cl)
                    except ValueError:
                        recall = np.nan
                    try:
                        f1 = f1_score(truth, prediction_cl)
                    except ValueError:
                        f1 = np.nan
                    try:
                        tn, fp, _, _ = confusion_matrix(truth, prediction_cl, labels=[0, 1]).ravel()
                        specificity_score = round(tn/(tn+fp), 3)
                    except ValueError:
                        specificity_score = np.nan
                    try:
                        roc_auc = roc_auc_score(truth, prediction)
                    except ValueError:
                        roc_auc = np.nan

                    if j == 0:
                    # save MCCV scores
                        test_score_row = pd.DataFrame({
                            'random_state': [random_numbers[i]],
                            'precision': [precision],
                            'recall': [recall],
                            'specificity': [specificity_score],
                            'f1': [f1],
                            'roc_auc': [roc_auc],
                            'accuracy': [accuracy]
                        })
                    else:
                        train_score_row = pd.DataFrame({
                            'random_state': [random_numbers[i]],
                            'precision': [precision],
                            'recall': [recall],
                            'specificity': [specificity_score],
                            'f1': [f1],
                            'roc_auc': [roc_auc],
                            'accuracy': [accuracy]
                        })

            # Regression model evaluation    
            else:

                for prediction, truth, j in zip(predictions, truths, [0,1]):
                    # Evaluation metrics
                    rmse = root_mean_squared_error(truth, prediction)
                    q2 = r2_score(truth, prediction)

                    if j == 0:
                        # save MCCV scores
                        test_score_row = pd.DataFrame({
                            'random_state': [random_numbers[i]],
                            'rmse': [rmse],
                            'q2': [q2]
                        })
                    else:
                        train_score_row = pd.DataFrame({
                            'random_state': [random_numbers[i]],
                            'rmse': [rmse],
                            'q2': [q2]
                        })    

            if len(scores) == 0:
                scores = test_score_row
                if return_train:
                    train_scores = train_score_row
            else:
                scores = pd.concat([scores, test_score_row], ignore_index=True)
                if return_train:
                    train_scores = pd.concat([train_scores, train_score_row], ignore_index=True)


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
        

    def mb_vip_permtest(self, x, y, n_permutations=1000, return_scores=False):
        """
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

        Args:
            x (array or list[array]): All blocks of predictors x1, x2, ..., xn. Rows are observations, columns are features/variables.
            y (array): 1-dim or 2-dim array of reference values, either continuous or categorical variable.
            n_permutations (int, optional): Number of permutation tests. Defaults to 1000.
            return_scores (bool, optional): Whether to return MB-VIP scores for each permuted null model. Defaults to False.

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

        # Fit Null models
        _vip_null = []
        for i in range(n_permutations):
            # Permute outcome
            y_perm = np.random.permutation(_y)
            # Fit permuted model and calculate MB-VIP scores
            self.fit(x, y_perm)
            _vip_null.append(self.mb_vip(plot=False, get_scores=True))
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
    def group_train_test_split(x, y, gropus=None, test_size=0.2, random_state=42):
        """
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
        Private Method
        Static Method
        
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
