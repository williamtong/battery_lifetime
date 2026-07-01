import shap


def create_shape_explanation(raw_model, X, y, columns_of_interest=None):
    '''
    Computes SHAP values for a trained tree model using shap.TreeExplainer.
    Returns a shap.Explanation object which can be passed directly to shap
    summary, waterfall, and dependence plots.

    Note: the columns_of_interest parameter is accepted for interface consistency
    but does not currently filter the SHAP computation — all columns in X are
    always explained. Filtering by column should be done on the returned
    explanation object if needed.

    Input:
    raw_model:                    Trained scikit-learn or XGBoost model (not wrapped
                                  in Tree_Model — pass model.model if using the wrapper).
    X (pandas DataFrame):         Feature matrix to explain, same format as used for
                                  training. Typically the training set.
    y (pandas Series or array):   Target values corresponding to X.
    columns_of_interest (list):   Currently unused. Pass None (default) to explain
                                  all columns.

    Output:
    explanation (shap.Explanation): SHAP explanation object containing values,
                                    base values, and feature names. Compatible with
                                    all standard shap plotting functions.
    '''
    if columns_of_interest is None:
        columns_of_interest = X.columns

    explainer = shap.TreeExplainer(raw_model)
    explanation = explainer(X=X,
                            y=y,
                            # check_additivity=False suppresses a warning that fires
                            # when SHAP values don't sum exactly to the model output,
                            # which can happen with tree ensembles due to floating point.
                            check_additivity=False)
    return explanation