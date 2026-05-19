import numpy as np
import random
import math
from itertools import product
from multiprocessing import Pool
import time

# For turbo-speeding using Numba 
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

# The main function of the program that searches for a solution
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

    # Tracking current number of queens on the board
    current_y_sum = 0 

    total_neurons = N * N

    for era in range(max_era):
        
        decay_factor = 0.998 ** era
        current_alpha_lr = alpha_learn_rate * decay_factor
        current_beta_lr = beta_learn_rate * decay_factor

        for _ in range(total_neurons):
            idx = random.randint(0, total_neurons - 1)
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
            noise = random.gauss(0, noise_std)
            delta = -(l - 1) - (c - 1) + rho_l + rho_c - 0.9 * (d1 - y[i, j]) - 0.9 * (d2 - y[i, j]) + noise
            old_x = x[i, j]
            x[i, j] += delta * step
            

            # Updating parameters for hysteresis if enabled
            if use_hysteresis:
                g = gamma_mat[i, j]
                abs_delta = abs(delta)

                if last_dx[i, j] >= 0:
                    cosh_arg = g * (x[i, j] + alpha[i, j])
                    cosh_arg = max(-50.0, min(cosh_arg, 50.0))
                    grad = g / (math.cosh(cosh_arg) ** 2) 
                
                    alpha[i, j] += current_alpha_lr * abs_delta * grad - noise
                    alpha[i, j] = min(alpha[i, j], alpha_max)
            
                else:
                    cosh_arg = g * (x[i, j] - beta[i, j])
                    cosh_arg = max(-50.0, min(cosh_arg, 50.0))
                    grad = g / (math.cosh(cosh_arg) ** 2)
                
                    beta[i, j] += current_beta_lr * abs_delta * grad - noise
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

            # Updating gamma locally according to the state of neuron
            if y[i, j] == 1:
                if l == 1 and c == 1 and d1 == 1 and d2 == 1:
                    gamma_mat[i, j] = min(gamma_max, gamma_mat[i, j] * 1.05)
                else: gamma_mat[i, j] = gamma_init
            else: gamma_mat[i, j] = gamma_init

            # Checking the solution
            if current_y_sum == N:
                # For a valid solution checking whether there are any conflicts in system and exit if not
                if get_conflicts(N, line_sum, col_sum, diag1_sum, diag2_sum) == 0:
                    return y, era                
        
    return None, max_era

# Running test with best parameters found
if __name__ == "__main__":
    start_time = time.time()

    # Final test: HHNN (With hysteresis)
    HHNN_success = 0
    for _ in range(5000):
        sol, _ = solve_nqueens(30, use_hysteresis=True, alpha_learn_rate=0.004, beta_learn_rate=0.002)
        if sol is not None:
            HHNN_success += 1
    print(f"Success rate of HHNN (5000 tests): {HHNN_success / 5000 * 100:.3f}%")
    HHNN_time = time.time()
    print(f"Total execution time: {HHNN_time - start_time:.2f} s.")

    # Final test: HNN (Basic)
    HNN_success = 0
    for _ in range(5000):
        sol, _ = solve_nqueens(30, use_hysteresis=False)
        if sol is not None:
            HNN_success += 1
    print(f"Success rate of HNN (5000 tests): {HNN_success / 5000 * 100:.3f}%")
    
    print(f"Total execution time: {time.time()-HHNN_time:.2f} s")
