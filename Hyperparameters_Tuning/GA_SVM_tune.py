import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA,MixedVariablePSO
from pymoo.termination import get_termination
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from pymoo.core.variable import Real, Integer, Choice, Binary




X_train = pd.read_csv('X_train_FS.csv', index_col=0).values
y_train = pd.read_csv('Y_train_FS.csv', index_col=0).values.flatten()
X_test = pd.read_csv('X_test_FS.csv', index_col=0).values
y_test = pd.read_csv('Y_test_FS.csv', index_col=0).values.flatten()



class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "kernel": Choice(options=["rbf", "sigmoid", "poly","linear"]),
            "C": Real(bounds=(0.1, 500)),
            "gamma": Choice(options=["scale", "auto"]),
            "degree": Integer(bounds=(1, 100)),
        }
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        kernel, C, gamma, degree = X["kernel"],  X["C"],X["gamma"], X["degree"]

        svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        scores = cross_val_score(svm, X_train, y_train, cv=10, scoring='f1')
        mean_f1_score = np.mean(scores)


        out["F"] = -mean_f1_score


problem = MixedVariableProblem()

algorithm = MixedVariableGA(pop=25)

res = minimize(problem,
               algorithm,
               termination=('n_evals', 1000),
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))