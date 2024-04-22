from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mixed import MixedVariableGA
from pymoo.termination import get_termination
from pymoo.core.variable import Real, Integer, Choice, Binary

X_train = pd.read_csv('X_train_FS.csv', index_col=0).values
y_train = pd.read_csv('Y_train_FS.csv', index_col=0).values.flatten()
X_test = pd.read_csv('X_test_FS.csv', index_col=0).values
y_test = pd.read_csv('Y_test_FS.csv', index_col=0).values.flatten()

class MixedVariableProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        vars = {
            "activation": Choice(options=["identity", "logistic", "tanh", "relu"]),
            "alpha": Real(bounds=(0.0001, 0.9)),
            "hidden_layer_sizes": Integer(bounds=(1, 100)),
            "learning_rate": Choice(options=["constant", "invscaling", "adaptive"]),
            "solver": Choice(options=["lbfgs", "sgd", "adam"]),
        }
        super().__init__(vars=vars, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        activation, alpha, hidden_layer_sizes, learning_rate, solver = X["activation"],  X["alpha"], X["hidden_layer_sizes"], X["learning_rate"], X["solver"]

        mlp = MLPClassifier(max_iter=4000,activation=activation, alpha=alpha, hidden_layer_sizes=(hidden_layer_sizes,), learning_rate=learning_rate, solver=solver, random_state=1)
        scores = cross_val_score(mlp, X_train, y_train, cv=10, scoring='f1')
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