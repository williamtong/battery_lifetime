from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error

import importlib
import matplotlib.pyplot as plt
import numpy as np

# Font size for parity plot axis labels and titles.
parity_fontsize = 18

# --- Default model hyperparameters ---
# These are passed into Tree_Model via the params argument.
# rfm_params: Random Forest Model (scikit-learn).
rfm_params = {'n_estimators': 80,
              'max_features': 0.5,
              'max_depth': 300,
              'bootstrap': True,
              'criterion': "squared_error",
              'verbose': 1,
              'oob_score': True,
              'n_jobs': 8
              }

# xgboost_params: Gradient Boosting Model (XGBoost).
# n_estimators is set very high; early_stopping_rounds controls the actual stopping point.
xgboost_params = {'n_estimators': 1000000,
                  'max_depth': 6,
                  'verbosity': 3,
                  'n_jobs': 8,
                  'eval_metric': 'rmse',
                  'colsample_bytree': 0.5,
                  'colsample_bynode': 0.5,
                  'learning_rate': 0.0025,
                  'verbose': 1000,  # Print eval error every 1000 trees.
                  'early_stopping_rounds': 20
                  }


def get_y_w_threshold(df, threshold):
    '''
    Counts the number of values in each column of df that exceed the threshold.
    Used to convert a matrix of per-cycle capacity values into a single target
    variable: the number of cycles until the battery crosses the SOH threshold.

    Input:
    df (pandas DataFrame):  Per-cycle measurements, shape (n_cycles, n_batteries).
    threshold (float):      The SOH threshold to count against.

    Output:
    df_above_threshold (pandas Series): Count of cycles exceeding threshold per battery.
    '''
    df_above_threshold = np.sum(df > threshold, axis=0)
    return df_above_threshold


class Tree_Model():
    '''
    A wrapper class for scikit-learn Random Forest or XGBoost tree models.
    Provides a consistent fit/predict interface with automatic error reporting
    and parity plot visualization after training and prediction.
    '''

    def __init__(self, Model, params, cycle_max=2500, random_state=1, plot_training=True):
        '''
        Instantiates and configures the tree model.

        Input:
        Model (class):          A scikit-learn or XGBoost model class
                                (e.g. RandomForestRegressor, XGBRegressor).
        params (dict):          Hyperparameters to pass to the model via set_params().
        cycle_max (int/float):  Upper axis limit for parity plots. For plotting only —
                                does not affect model training or data filtering.
        random_state (int):     Random seed for reproducibility.
        plot_training (bool):   If True, plots a parity plot after fitting.
        '''
        self.cycle_max = cycle_max  # For plotting only.
        self.params = params
        self.model = Model(random_state=random_state)
        self.model.set_params(**self.params)
        self.plot_training = plot_training
        print('Tree_Model initiated.')

    def fit(self, X_train, y_train_actual):
        '''
        Trains the model on the provided data. Detects whether the model is an
        XGBoost GBM (via the presence of 'early_stopping_rounds' in params) or
        a scikit-learn RFM, and calls the appropriate fit signature.
        Prints training error metrics and optionally plots a parity plot.

        Input:
        X_train (pandas DataFrame):            Training features.
        y_train_actual (pandas Series or 1D array): Training target variable
                                               (number of cycles to SOH threshold).

        Output:
        model: The trained model object.
        '''
        self.y_train_actual_type = type(y_train_actual)
        y_train_actual = y_train_actual.values

        # XGBoost requires eval_set and verbose to be passed directly to fit().
        # scikit-learn models do not accept these arguments.
        if 'early_stopping_rounds' in self.params.keys():
            self.model.fit(X_train, y_train_actual,
                           verbose=self.params['verbose'],
                           eval_set=self.params['eval_set'])
        else:
            self.model.fit(X_train, y_train_actual)

        # Calculate and report training error (not holdout — for diagnostic purposes only).
        print("Finished fitting.  Predicting X...")
        y_train_predict = self.model.predict(X_train)
        print("Finished predicting X.")
        self.MdAPE_train = np.median(np.abs(y_train_predict - y_train_actual)/y_train_actual)*100
        print(f'Training MdAPE is {self.MdAPE_train:.3f}% (not holdout).')

        # Store feature importances sorted in descending order for later inspection.
        self.feature_imp = sorted(
            [feature for feature in zip(X_train.columns, self.model.feature_importances_)],
            key=lambda feature: feature[1], reverse=True)
        print("Finished training model.")

        error_calculation_output(y_train_actual, y_train_predict)

        if self.plot_training:
            plot_scatter_plot(y_train_actual,
                              y_train_predict,
                              "Training data set",
                              cycle_max=self.cycle_max,
                              color='r')

        return self.model

    def predict(self, X, y_actual=None):
        '''
        Runs inference on X. If actual labels are supplied, prints error metrics
        and plots a parity plot of predicted vs actual values.

        Input:
        X (pandas DataFrame):               Features to predict.
        y_actual (pandas Series or None):   Actual target values. If provided,
                                            error metrics and a parity plot are produced.

        Output:
        y_predict (array): Predicted cycle counts.
        '''
        y_predict = self.model.predict(X)

        # If y_actual is provided and matches the training data type, report holdout errors.
        y_actual_data_type = type(y_actual)
        if y_actual_data_type == self.y_train_actual_type:
            y_actual = y_actual.values

            error_calculation_output(y_actual, y_predict)

            # Plot parity plot for holdout results.
            plot_scatter_plot(y_actual,
                              y_predict,
                              "Holdout data set",
                              cycle_max=self.cycle_max,
                              color='b')

        return y_predict


def plot_scatter_plot(y_actual, y_pred, title="Holdout data set", cycle_max=300, color='r'):
    '''
    Plots a parity plot (predicted vs actual) for regression model evaluation.
    A diagonal reference line is included so deviations from perfect prediction
    are immediately visible.

    Input:
    y_actual (array):   Actual cycle counts.
    y_pred (array):     Predicted cycle counts.
    title (str):        Plot title.
    cycle_max (int):    Upper axis limit for both axes.
    color (str):        Scatter point color.
    '''
    print("Plotting scatter plot...")
    plt.figure(figsize=(12, 12))
    plt.title(title, fontsize=parity_fontsize + 4)
    # Plot a faint diagonal reference line representing perfect prediction.
    plt.scatter(range(0, cycle_max, 20), range(0, cycle_max, 20),
                marker='o', s=1, alpha=0.2, color=color)
    plt.scatter(y_actual, y_pred, alpha=1, color=color)
    plt.xlabel("degradation cycles actual", fontsize=parity_fontsize)
    plt.ylabel("degradation cycles predicted", fontsize=parity_fontsize)
    plt.grid()
    plt.xlim(0, cycle_max)
    plt.ylim(0, cycle_max)
    plt.xticks(fontsize=parity_fontsize)
    plt.yticks(fontsize=parity_fontsize)
    plt.show()


def error_calculation_output(y_actual, y_predict):
    '''
    Prints MdAPE, MAPE, RMSE, MAE, and R² for a set of predictions.
    MdAPE (Median Absolute Percentage Error) is the primary metric for this
    project because it is robust to outliers and directly interpretable as a
    percentage of the true cycle count.

    Input:
    y_actual (array):   Actual cycle counts.
    y_predict (array):  Predicted cycle counts.
    '''
    MdAPE = np.median(np.abs(y_predict - y_actual)/y_actual)*100
    MAPE = np.mean(np.abs(y_predict - y_actual)/y_actual)*100
    print(f'MdAPE is {MdAPE:.3f}%, MAPE is {MAPE:.3f}%')
    rmse = np.sqrt(mean_squared_error(y_actual, y_predict))
    mae = mean_absolute_error(y_actual, y_predict)
    score = r2_score(y_actual, y_predict)
    print("R2: %5.3f, RMSE: %5.3f, MAE: %5.3f" % (score, rmse, mae))