

from itertools import product

def GridSearch(estimator, param_grid, X, y, cv=None):
    avg_scores = {}
    # iterate over the product of parameter values
    for val in product(*param_grid.values()):
        # recreate a dictionary with a single value for each parameter
        params = dict(zip(param_grid.keys(), val))
        # feed the estimator with the current parameters
        model = estimator(**params)
        # cross validation
        cv_scores = cross_val_score(model, X, y, cv=cv)
        # generate a string with the parameter names and their value
        key = '_'.join([k+'_'+str(v) for k, v in params.items()])
        # store cv_score in dictionary
        avg_scores[key] = np.average(cv_scores)
    return avg_scores


GridSearch(estimator = SVC,
           param_grid = {'kernel':['linear', 'rbf'], 'C':[1, 10]},
           X = X_train,
           y = y_train,
           cv = 5)
