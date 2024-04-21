import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.termination import get_termination
from xgboost import XGBClassifier




X_train = pd.read_csv('X_train_FS.csv', index_col=0).values
y_train = pd.read_csv('Y_train_FS.csv', index_col=0).values.flatten()
X_test = pd.read_csv('X_test_FS.csv', index_col=0).values
y_test = pd.read_csv('Y_test_FS.csv', index_col=0).values.flatten()

class FeatureSelectionProblem(ElementwiseProblem):
    def __init__(self):
        n_features = X_train.shape[1]
        super().__init__(n_var=n_features, n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        threshold = 0.75  # Adjust this threshold as needed
        selected_features = X > threshold
        clf = XGBClassifier()
        clf.fit(X_train[:, selected_features], y_train)
        y_pred = clf.predict(X_test[:, selected_features])
        accuracy = f1_score(y_test, y_pred)
        out["F"] = -accuracy

# Initialize optimization problem
problem = FeatureSelectionProblem()

# Initialize GA algorithm
algorithm = PSO(pop_size=25)

# Define termination criteria
termination = get_termination("n_eval", 1000)

# Perform optimization
res = minimize(problem, algorithm, termination,  seed=1, verbose=True)

# Get best solution
best_solution = res.X
value = -res.F
print("best solution", best_solution)
best_solution = best_solution > 0.75
selected_features = np.where(best_solution)[0]
print("Selected Features:", selected_features)

# Save selected features to CSV file
selected_features_df = pd.DataFrame(selected_features, columns=["Selected_Features"])
selected_features_df.to_csv("selected_featuresPSO.csv", index=False)
print("best result",value)