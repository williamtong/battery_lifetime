import shap

def create_shape_explanation(raw_model, X, y, columns_of_interest = None):  # None means all columns are of interest
    '''
    input:
    raw_model: trained xgboost or scikit-learn model (no wrapper)
    X (pandas dataframe): X data same format to train raw_model, usually the training data set.
    y (1-D pandas dataframe or np array): y data, same as X.
    columns_of_interest:  Note: None means all columns are of interest

    output: shap.explanation object
    '''
    if columns_of_interest is None:
        columns_of_interest = X.columns
        
    explainer = shap.TreeExplainer(raw_model)
    # shap_values = explainer.shap_values(X)
    explanation = explainer(X = X,
                            y = y, 
                            # feature_perturbation = 'interventional',
                            check_additivity=False 
                           )
    return explanation