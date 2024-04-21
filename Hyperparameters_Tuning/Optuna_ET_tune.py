import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.termination import get_termination
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from pymoo.core.variable import Real, Integer, Choice, Binary




X_train = pd.read_csv('X_train_FS.csv', index_col=0).values
y_train = pd.read_csv('Y_train_FS.csv', index_col=0).values.flatten()
X_test = pd.read_csv('X_test_FS.csv', index_col=0).values
y_test = pd.read_csv('Y_test_FS.csv', index_col=0).values.flatten()




class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "bootstrap": Binary(),
            "criterion": Choice(options=["gini", "entropy", "log_loss"]),
            "max_depth": Integer(bounds=(1, 200)),
            "max_features": Integer(bounds=(1, 300)),
            "n_estimators": Integer(bounds=(1, 300)),
            
        }
    
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        bootstrap, criterion, max_depth, max_features, n_estimators = X["bootstrap"], X["criterion"], X["max_depth"], X["max_features"], X["n_estimators"]

        clf = ExtraTreesClassifier(bootstrap=bootstrap, criterion=criterion, max_depth=max_depth,
                                   max_features=max_features, n_estimators=n_estimators, random_state=1)
        scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1')
        mean_f1_score = np.mean(scores)


        out["F"] = -mean_f1_score


problem = MixedVariableProblem()

algorithm = Optuna(pop=25)

res = minimize(problem,
               algorithm,
               termination=('n_evals', 1000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

