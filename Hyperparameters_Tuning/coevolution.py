import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from deap import base, creator, tools, algorithms
import random

# Load the training and testing data from CSV files
X_train = pd.read_csv('X_train_FS.csv')
X_test = pd.read_csv('X_test_FS.csv')
y_train = pd.read_csv('Y_train_FS.csv').values.ravel()
y_test = pd.read_csv('Y_test_FS.csv').values.ravel()

# Define the problem type (maximize F1 score of meta-SVM)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define parameter space for SVM
param_space_svm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': list(range(1, 21)),  # Degree range 1 to 20
    'C': list(np.linspace(1.0, 20.0, num=100)),  # C range 1.0 to 20.0
    'gamma': ['auto', 'scale']
}

# Define parameter space for MLP
param_space_mlp = {
    'learning_rate_init': list(np.linspace(0.0001, 0.5, num=100)),  # Learning rate init range 0.0001 to 0.5
    'learning_rate': ['constant', 'adaptive'],
    'hidden_layer_sizes': [(50, 50, 50), (100,), (150,), (200,)],  # Hidden layer sizes options
    'activation': ['relu', 'sigmoid']
}

# Define parameter space for ExtraTree
param_space_extra = {
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(10, 101)),  # Max depth range 10 to 100
    'max_features': list(range(10, 201)),  # Max features range 10 to 200
    'n_estimators': [200]
}

# Define functions to initialize individuals for all species within the specified ranges
def init_individual_svm():
    individual = [random.choice(param_space_svm[key]) for key in param_space_svm.keys()]
    return individual

def init_individual_mlp():
    individual = [random.choice(param_space_mlp[key]) for key in param_space_mlp.keys()]
    return individual

def init_individual_extra():
    individual = [random.choice(param_space_extra[key]) for key in param_space_extra.keys()]
    return individual

# Define mutation functions to ensure hyperparameters remain within constraints
def mutate_svm(individual, mutation_prob=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.choice(param_space_svm[list(param_space_svm.keys())[i]])
    return individual,

def mutate_mlp(individual, mutation_prob=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.choice(param_space_mlp[list(param_space_mlp.keys())[i]])
    return individual,

def mutate_extra(individual, mutation_prob=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = random.choice(param_space_extra[list(param_space_extra.keys())[i]])
    return individual,

# Define evaluation function to calculate joint fitness (F1 score of meta-SVM)
def evaluate(individuals):
    svm_rep = individuals[0]
    mlp_rep = individuals[1]
    extra_rep = individuals[2]

    # Extract hyperparameters
    svm_kernel, svm_degree, svm_C, svm_gamma = svm_rep
    mlp_learning_rate_init, mlp_learning_rate, mlp_hidden_layer_sizes, mlp_activation = mlp_rep
    extra_bootstrap, extra_criterion, extra_max_depth, extra_max_features, extra_n_estimators = extra_rep

    # Train SVM classifier with 5-fold cross-validation
    clf_svm = SVC(kernel=svm_kernel, degree=svm_degree, C=svm_C, gamma=svm_gamma)
    svm_scores = cross_val_score(clf_svm, X_train, y_train, cv=5, scoring='f1_weighted')
    svm_score = svm_scores.mean()

    # Train MLP classifier with 5-fold cross-validation
    clf_mlp = MLPClassifier(learning_rate_init=mlp_learning_rate_init,
                            learning_rate=mlp_learning_rate,
                            hidden_layer_sizes=mlp_hidden_layer_sizes,
                            activation=mlp_activation)
    mlp_scores = cross_val_score(clf_mlp, X_train, y_train, cv=5, scoring='f1_weighted')
    mlp_score = mlp_scores.mean()

    # Train ExtraTree classifier with 5-fold cross-validation
    clf_extra = ExtraTreesClassifier(bootstrap=extra_bootstrap,
                                    criterion=extra_criterion,
                                    max_depth=extra_max_depth,
                                    max_features=extra_max_features,
                                    n_estimators=extra_n_estimators)
    extra_scores = cross_val_score(clf_extra, X_train, y_train, cv=5, scoring='f1_weighted')
    extra_score = extra_scores.mean()

    # Concatenate prediction probabilities
    X_test_concat = np.concatenate((X_test, svm_prob, mlp_prob, extra_prob), axis=1)

    # Use SVM to predict on concatenated set
    clf_meta = SVC(kernel='rbf', C=1.7, gamma='auto')
    clf_meta.fit(X_test_concat, y_test)
    y_pred_meta = clf_meta.predict(X_test_concat)

    # Calculate F1 score of meta-SVM
    f1_meta = f1_score(y_test, y_pred_meta, average='weighted')

    return f1_meta,

# Create a toolbox for the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("individual_svm", tools.initIterate, creator.Individual, init_individual_svm)
toolbox.register("individual_mlp", tools.initIterate, creator.Individual, init_individual_mlp)
toolbox.register("individual_extra", tools.initIterate, creator.Individual, init_individual_extra)
toolbox.register("mutate_svm", mutate_svm)
toolbox.register("mutate_mlp", mutate_mlp)
toolbox.register("mutate_extra", mutate_extra)
toolbox.register("evaluate", evaluate)

# Define the number of species and individuals
NUM_SPECIES = 3
NUM_INDIVIDUALS = 10

# Create species for SVM, MLP, and ExtraTree
species = [
    [toolbox.individual_svm() for _ in range(NUM_INDIVIDUALS)],
    [toolbox.individual_mlp() for _ in range(NUM_INDIVIDUALS)],
    [toolbox.individual_extra() for _ in range(NUM_INDIVIDUALS)]
]

# Select representatives from each species
representatives = [random.choice(species[i]) for i in range(NUM_SPECIES)]

# Define number of generations
num_generations = 30

# Run the coevolutionary genetic algorithm
for gen in range(num_generations):
    next_repr = [None] * len(species)
    for (i, s) in enumerate(species):
        # Vary the species individuals
        s = algorithms.varAnd(s, toolbox, 0.6, 1.0)

        # Get the representatives excluding the current species
        r = representatives[:i] + representatives[i+1:]
        for ind in s:
            # Evaluate and set the individual fitness
            ind.fitness.values = toolbox.evaluate([ind] + r)

        # Select the individuals
        species[i] = toolbox.select(s, len(s))  # Tournament selection
        next_repr[i] = tools.selBest(s, k=1)[0]   # Best selection

    representatives = next_repr

# Get the best individual for SVM, MLP, and ExtraTree
best_svm = tools.selBest(species[0], k=1)[0]
best_mlp = tools.selBest(species[1], k=1)[0]
best_extra = tools.selBest(species[2], k=1)[0]

print("Best SVM parameters:", best_svm)
print("Best MLP parameters:", best_mlp)
print("Best ExtraTree parameters:", best_extra)
