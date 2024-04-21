import numpy as np
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import math

# Define the function to calculate fitness
def func(data, input_features):
    if len(input_features) == 0:
        return 1  # Return a high error if no features are selected
    else:
        X = data[:, input_features]
        y = data[:, -1]
        knn = KNeighborsClassifier(n_neighbors=4)
        # Perform 10-fold cross-validation and calculate the mean error rate
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        err = 1 - scores.mean()  # Error is the complement of accuracy
        return err

# Load data
data = np.load('sonar.npy')

# Dimensionality
d = data.shape[1] - 1

# Maximal number of fitness evaluations
maxfe = 100

# Number of trial runs
runnum = 30

# Population size
m = 50

# Bounds
lu = np.array([[-5] * d, [5] * d])

# Initialize fitness matrix
fitness = np.zeros((m, 1))

# Set phi
phi = 0.1

# Initialize results array
results = np.zeros(runnum)

# Multiple runs
for run in range(runnum):
    # Initialization
    XRRmin = np.tile(lu[0], (m, 1))
    XRRmax = np.tile(lu[1], (m, 1))
    np.random.seed(sum(100 * np.random.randn(1)))
    p = XRRmin + np.random.rand(m, d) * (XRRmax - XRRmin)
    bi_position = np.zeros((m, d))
    pop = expit(p)
    RandNum = np.random.rand(m, d)
    change_position = (pop > RandNum)
    bi_position[change_position] = 1
    
    # Calculate fitness for each individual
    for i in range(m):
        feature = np.where(bi_position[i] == 1)[0]
        fitness[i, 0] = func(data, feature)
    
    v = np.zeros((m, d))
    bestever = 1e200
    gen = 0
    
    # Main loop
    while gen < maxfe:
        # Generate random pairs
        rlist = np.random.permutation(m)
        midpoint = int(np.ceil(m / 2))
        rpairs = np.vstack((rlist[:midpoint], rlist[midpoint:])).T
        
        # Calculate center position
        center = np.ones((int(np.ceil(m/2)), 1)) * np.mean(p, axis=0)

        
        # Do pairwise competitions
        mask = fitness[rpairs[:, 0]] > fitness[rpairs[:, 1]]
        losers = np.where(mask, rpairs[:, 0], rpairs[:, 1])
        winners = np.where(mask, rpairs[:, 1], rpairs[:, 0])
        
        # Generate random matrices
        randco1 = np.random.rand(int(np.ceil(m/2)), d)
        randco2 = np.random.rand(int(np.ceil(m/2)), d)
        randco3 = np.random.rand(int(np.ceil(m/2)), d)

        
        # Losers learn from winners
        v[losers, :] = randco1 * v[losers, :] + randco2 * (p[winners, :] - p[losers, :]) + phi * randco3 * (center - p[losers, :])
        p[losers, :] += v[losers, :]
        
        # Boundary control
        for i in range(1, math.ceil(m / 2) + 1):
         loser_index = losers[i - 1]
         p[loser_index] = np.maximum(p[loser_index], lu[0])
         p[loser_index] = np.minimum(p[loser_index], lu[1])

        
        # Fitness evaluation
        # Fitness evaluation
        n_losers = len(losers)
        for id in range(n_losers):
            pop[losers[id]] = expit(p[losers[id]])
            Randnu = np.random.rand(1, n)
            change_pos = pop[losers[id]] > Randnu
            feature = np.where(change_pos)[0]
            fitness[losers[id], 0] = func(data, feature)

        
        bestever = min(bestever, np.min(fitness))
        print(f'Runs: {run+1}\t Iter: {gen}\t Best fitness: {bestever:.4f}')
        gen += 1
    
    results[run] = bestever
    print(f'Run No.{run+1} Done!')
