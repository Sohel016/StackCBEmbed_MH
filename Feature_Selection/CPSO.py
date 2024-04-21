import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def initialize_particle_position(D):
    return [random.uniform(0, 1) for _ in range(D)]


# Function to calculate the median fitness value of a population P
def median_fitness_value_of_P(P):
    fitness_values = [f(X) for X in P]
    sorted_fitness_values = sorted(fitness_values)
    median_index = len(sorted_fitness_values) // 2
    return sorted_fitness_values[median_index]

# Function to find candidates better than a given threshold gamma_f
def candidates_better_than_gamma_f(P, gamma_f):
    return [X for X in P if f(X) > gamma_f]

# Function to calculate flipping rates of X's bits based on Eqs. (6) and (7)
def calculate_flipping_rates(X):
    D = len(X)
    D0 = X.count(0)
    D1 = X.count(1)
    si = [0.5] * D  # Placeholder for si values, replace with actual relevant scores
    flipping_rates = []
    for i in range(D):
        if X[i] == 0:
            flipping_rate = D0 / D * si[i] / sum(si[:D0])
        else:
            flipping_rate = D1 * 10 / D * (1 - si[i]) / sum(1 - si[D0:])
        flipping_rates.append(flipping_rate)
    return flipping_rates

# Function to randomly flip bits of X
def randomly_flip_bits(X):
    flipping_rates = calculate_flipping_rates(X)
    return [1 if random.random() < rate else 0 for rate in flipping_rates]

# Function to evaluate the fitness of a feature subset
def fitness_function(S, alpha, X_train, X_test, y_train, y_test):
    selected_features = [i for i in range(len(S)) if S[i] == 1]
    X_train_subset = X_train[:, selected_features]
    X_test_subset = X_test[:, selected_features]
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train_subset, y_train)
    y_pred = knn_classifier.predict(X_test_subset)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    subset_size_ratio = sum(S) / len(S)
    return alpha * error_rate + (1 - alpha) * subset_size_ratio

# Algorithm 1: Local Search on the Top 10% Population
def local_search(P_top, m, alpha, X_train, X_test, y_train, y_test):
    for X in P_top:
        flipping_rates = calculate_flipping_rates(X)
        X_n = None
        for l in range(10 * m):
            X_n = randomly_flip_bits(X)
            if fitness_function(X_n, alpha, X_train, X_test, y_train, y_test) < fitness_function(X, alpha, X_train, X_test, y_train, y_test):
                # Found a potentially better solution than X
                X = X_n
                break

# Algorithm 2: Size-Change Operator (Reinitialize Population)
def size_change_operator(S_best, m, alpha, X_train, X_test, y_train, y_test):
    NF_best = sum(S_best)
    P_new = []
    for i in range(m):
        NFi = random.randint(1, NF_best)
        X_i_best = None
        for k in range(m):
            # Implement roulette wheel selection based on Relief scores
            X_i_tmp = subset_selected_by_roulette_wheel_selection_based_on_relief_scores()
            if fitness_function(X_i_tmp, alpha, X_train, X_test, y_train, y_test) < fitness_function(X_i_best, alpha, X_train, X_test, y_train, y_test):
                X_i_best = X_i_tmp
        P_new.append(X_i_best)
    return P_new

# Function to evaluate the fitness of a population
def evaluate_P(P, alpha, X_train, X_test, y_train, y_test):
    for X in P:
        fitness = fitness_function(X, alpha, X_train, X_test, y_train, y_test)
        # Evaluate fitness and store it somewhere

# Function to update the surrogate training set
def update_SurTrainingSet(SurTrainingSet, P_next):
    for i in range(0, len(P_next), 2):
        X1 = P_next[i]
        X2 = P_next[i+1]
        SurTrainingSet.append((X1, X2))

def train_surrogate_model(SurTrainingSet):
    X_train = [pair[0] for pair in SurTrainingSet]
    y_train = [pair[1] for pair in SurTrainingSet]
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Function to update the surrogate model and retrain if necessary
def update_surrogate_model(SurTrainingSet, SurFreq, iteration):
    if iteration % SurFreq == 0:
        svm_classifier = train_surrogate_model(SurTrainingSet)
        return svm_classifier
    else:
        return None  # No need to retrain


def randomly_pick_from_FeaSet(FeaSet):
    # Randomly select two distinct indices
    idx1, idx2 = random.sample(range(len(FeaSet)), 2)
    # Return the corresponding solutions
    return FeaSet[idx1], FeaSet[idx2]    

# Function to return the selected features based on the best particle
def subset_selected_by_X_best(X_best):
    return [i for i, val in enumerate(X_best) if val == 1]

# Algorithm 3: Overall Algorithm CCSO
def CCSO(ME, m, L, SC, alpha, X_train, X_test, y_train, y_test):
    eval = 0
    D = X_train.shape[1]  # Number of features
    P = [initialize_particle_position(D) for _ in range(m)]
    evaluate_P(P, alpha, X_train, X_test, y_train, y_test)
    eval += m
    X_best = None
    SurTrainingSet = []
    iteration = 0
    SurFreq = 40 * m  # Set SurFreq to 40 * m
    L_counter = 0  # Counter for tracking how many evaluations X_best has not changed
    SC_counter = 0  # Counter for tracking how many evaluations X_best has not changed

    while eval < ME:
        gamma_f = median_fitness_value_of_P(P)
        FeaSet = candidates_better_than_gamma_f(P, gamma_f)
        P_next = []
        for X1, X2 in pairs_from_P(P):
            X_w = arg_min(f(X1), f(X2))
            X_l = arg_max(f(X1), f(X2))
            update_SurTrainingSet(SurTrainingSet, P_next)
            if f(X_w) <= gamma_f:
                X_l_n = update_learner(X_l, X_w)
                X_w_n = X_w
                eval += 1
            else:
                X_fea1, X_fea2 = randomly_pick_from_FeaSet(FeaSet)
                X_l_n = update_learner(X_l, X_fea1)
                X_w_n = update_learner(X_w, X_fea2)
                eval += 2
            P_next.append(X_l_n)
            P_next.append(X_w_n)
        evaluate_P(P_next, alpha, X_train, X_test, y_train, y_test)
        X_best_next= best_particle(X_best, P_next)
        if iteration % SurFreq == 0:
            SurM = update_surrogate_model(SurTrainingSet, SurFreq, iteration)

        if X_best_next == X_best:
            L_counter += 1
            SC_counter += 1
        else:
            L_counter = 0
            SC_counter = 0
            X_best = X_best_next

        if L_counter >= L:
            local_search(P_next, SurM, m, alpha, X_train, X_test, y_train, y_test)

            eval += 0.1 * m
            L_counter = 0

        if SC_counter >= SC:
            P = size_change_operator(X_best, SurM, m, alpha, X_train, X_test, y_train, y_test)
            SC_counter = 0

        iteration += 1
    return subset_selected_by_X_best(X_best)

# Example usage:
# Define your training and testing data (X_train, X_test, y_train, y_test)
# Set algorithm parameters (ME, m, L, SC, alpha)
# Run the algorithm: selected_features = CCSO(ME, m, L, SC, alpha, X_train, X_test, y_train, y_test)
