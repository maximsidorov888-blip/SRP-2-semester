import numpy as np
import random
import math
from itertools import product
from multiprocessing import Pool
import time

# For turbo-speeding i'm using Numba 
from numba import njit

#Auxiliary function to calculate conflicts to each era
@njit
def get_conflicts(N, line_sum, col_sum, diag1_sum, diag2_sum):
    conflicts = 0
    for i in range(N):
        if line_sum[i] > 1: conflicts += (line_sum[i] - 1)
        if col_sum[i] > 1: conflicts += (col_sum[i] - 1)
    for j in range(2*N-1):
        if diag1_sum[j] > 1: conflicts += (diag1_sum[j] - 1)
        if diag2_sum[j] > 1: conflicts += (diag2_sum[j] - 1)
    return conflicts

@njit
def solve_nqueens(N, use_hysteresis=True, max_era=1500, step=0.2, alpha_learn_rate=0.01, beta_learn_rate=0.01,
                  alpha_max=1.3, beta_max=1.3, gamma_init=1.0, gamma_max=10.0, noise_std=0.001):
    
    # Matrices initialization 
    x = np.random.uniform(-0.5, 0.5, (N, N))
    y = np.zeros((N, N), dtype=np.int32)
    last_dx = np.random.choice(np.array([-1.0, 1.0]), (N, N))

    # Arrays to track sums for fast constraint checking
    line_sum = np.zeros(N, dtype=np.int32)
    col_sum = np.zeros(N, dtype=np.int32)
    diag1_sum = np.zeros(2*N-1, dtype=np.int32)
    diag2_sum = np.zeros(2*N-1, dtype=np.int32)

    # Parameters for hysteresis
    alpha = np.full((N, N), 0.01, dtype=np.float64)
    beta = np.full((N, N), 0.01, dtype=np.float64)
    gamma_mat = np.full((N, N), gamma_init, dtype=np.float64)

    # Tracking current_amount of conflicts and how long the system is stagnating
    best_conflicts_so_far = 999
    stagnation_counter = 0
    cooldown_counter = 0
    progress_score = 0

    # Tracking current number of queens on the board
    current_y_sum = 0 

    # Making an array of indices for more precise random selection of neurons.
    indices = np.arange(N * N)

    for era in range(max_era):

        np.random.shuffle(indices)  # Shuffle indices to ensure random selection of neurons
        for idx in indices:
            # Randomly selecting a neuron to update
            i = idx // N
            j = idx % N

            l = line_sum[i]
            c = col_sum[j]
            d1 = diag1_sum[i+j]
            d2 = diag2_sum[i-j+N-1]

            # Defining rho_l and rho_c based on whether the line or column is empty (0) or not (1)
            rho_l = 1 if l == 0 else 0
            rho_c = 1 if c == 0 else 0

            # Calculating the delta with added noise for better exploration of the solution space
            decay_factor = 0.998 ** era  # Exponential decay 
            noise = random.gauss(0, noise_std)
            delta = -(l - 1) - (c - 1) + rho_l + rho_c - (d1 - y[i, j]) - (d2 - y[i, j]) + noise
            old_x = x[i, j]
            x[i, j] += delta * step

            # Updating parameters for hysteresis if enabled
            if use_hysteresis:
                g = gamma_mat[i, j]
                abs_delta = abs(delta)

                current_alpha_lr = alpha_learn_rate * decay_factor
                current_beta_lr = beta_learn_rate * decay_factor
            
                if last_dx[i, j] >= 0:
                    cosh_arg = g * (x[i, j] + alpha[i, j])
                    cosh_arg = max(-50.0, min(cosh_arg, 50.0))
                    grad = g / (math.cosh(cosh_arg) ** 2) 
                
                    alpha[i, j] += current_alpha_lr * abs_delta * grad - 0.001
                    alpha[i, j] = min(alpha[i, j], alpha_max)
            
                else:
                    cosh_arg = g * (x[i, j] - beta[i, j])
                    cosh_arg = max(-50.0, min(cosh_arg, 50.0))
                    grad = g / (math.cosh(cosh_arg) ** 2)
                
                    beta[i, j] += current_beta_lr * abs_delta * grad - 0.001
                    beta[i, j] = min(beta[i, j], beta_max)
        
            # Updating the output of the neuron based on the current input and hysteresis thresholds
            if use_hysteresis:
                if last_dx[i, j] >= 0: 
                    new_y = 1 if x[i, j] > alpha[i, j] else 0
                else: 
                    new_y = 1 if x[i, j] > -beta[i, j] else 0
            else: 
                new_y = 1 if x[i, j] > 0 else 0

            # If the output has changed, we need to update the sums and the current count of queens on the board
            if new_y != y[i, j]:
                delta_y = new_y - y[i, j]
                line_sum[i] += delta_y
                col_sum[j] += delta_y
                diag1_sum[i+j] += delta_y
                diag2_sum[i-j+N-1] += delta_y
                y[i, j] = new_y
                current_y_sum += delta_y 
        
            # Updating the direction sign
            last_dx[i, j] = 1.0 if (x[i, j] - old_x) >= 0 else -1.0


            # Checking the solution
            if current_y_sum == N:
                # For a valid solution, there should be exactly 1 queen in each row/column,
                # and at most 1 queen on each diagonal.
                if (np.all(line_sum == 1) and np.all(col_sum == 1) and 
                    np.all(diag1_sum <= 1) and np.all(diag2_sum <= 1)):
                    return y, era

        # Updating gamma according to how close the system to solution is.
        conflicts = get_conflicts(N, line_sum, col_sum, diag1_sum, diag2_sum)
        max_possiblle_conflicts = N 
        score = 1.0 - conflicts / max_possiblle_conflicts
        progress_score = (current_y_sum / N) * max(0.0, score)
        
        # Tracking conflicts and stagnation time
        if conflicts > 0:
            if conflicts >= best_conflicts_so_far:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_conflicts_so_far = conflicts
            
            if stagnation_counter >= 50: #Making a shake-up for the system
                cooldown_counter = 20
                x += np.random.uniform(-0.1, 0.1, (N, N))
                stagnation_counter = 0
                best_conflicts_so_far = conflicts
        
        if cooldown_counter == 0:
            gamma_val = gamma_init + (gamma_max - gamma_init) * (progress_score ** 4) #The closer the system to the solution the bigger gamma become
            gamma_mat.fill(gamma_val)
        else:
            gamma_mat.fill(gamma_init)
            cooldown_counter -= 1                
        
    return None, max_era

# Function to evaluate a specific combination of learning rates by running multiple trials and calculating the success rate
def evaluate(params):
    alpha_lr, beta_lr = params
    N = 30
    trials = 1000
    success = 0
    for _ in range(trials):
        sol, _ = solve_nqueens(N, use_hysteresis=True, alpha_learn_rate=alpha_lr, beta_learn_rate=beta_lr)
        if sol is not None:
            success += 1
    rate = success / trials
    return (alpha_lr, beta_lr, rate)

# Launching the grid search for optimal learning rates and then testing the best configuration on a larger number of trials to evaluate the performance of both HHNN and HNN approaches.
if __name__ == "__main__":
    start_time = time.time()
    
    # Parameters for Grid Search
    N = 30
    a_lrs = np.arange(0.001, 0.01, 0.001)    
    b_lrs = np.arange(0.001, 0.01, 0.001)
    params_comb = list(product(a_lrs, b_lrs))
    print(f"Total combinations: {len(params_comb)}")

    # Calculating with multiprocessing to speed up the grid search
    with Pool(processes=8) as pool:
        results = pool.map(evaluate, params_comb)

    # Printing the results of the Grid Search
    best_params = None
    best_rate = 0
    for lr_a, lr_b, rate in results:
        print(f"Learning rates -> alpha: {lr_a}, beta: {lr_b} | Success rate: {rate*100:.2f}%")
        if rate > best_rate:
            best_rate = rate
            best_params = (lr_a, lr_b)
    
    print(f"\nBest rates: alpha = {best_params[0]}, beta = {best_params[1]}")
    print(f"Best success rate: {best_rate * 100:.2f}%\n")
    
    print(f"Total execution time: {time.time()-start_time:.2f} s")
