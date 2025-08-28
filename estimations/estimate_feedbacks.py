from relaxations import *
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# read experimental data
P = 2.2 

df_reversed = pd.read_csv("../data/insertion.csv",header=None)
df = df_reversed.iloc[::-1]
df = df.reset_index(drop=True)

# get indicies for comparison
t_before_data = (1110-df[0][0])*60
duration_data = (df.iloc[-1][0]-df[0][0])*60
t_end_data = df.iloc[-1][0]
t_before_sim = t_ins-t_before_data
T_insert = [t for t in T if (t > (t_before_sim)) and (t < (t_before_sim)+(duration_data))]
i_insert = [t[0] for t in enumerate(T) if (t[1] > (t_before_sim)) and (t[1] < (t_before_sim)+(duration_data))]

adj = (df[0][0])*60-T_insert[0]
df[0] = [(t*60)-adj for t in df[0]]

# adjust to reported power
d = df[1][0]-P
df[1] = [p-d for p in df[1]]

# Set up interpolation
# Assuming df[0] is time and df[1] is the data you want to interpolate
spline = CubicSpline(df[0], df[1])  # Multiplying df[0] by 60 if it's in minutes

# Use the spline to interpolate at the desired times
interpolated_values = spline(T[i_insert[0]:i_insert[-1]+1])

def sumSq_dP(params):
    try:
        # run
        sol_jit = relax_feedback(params)

        # calculate error
        simulation_output = np.array([P*(s[6]-sol_jit[i_insert[0]-1][6]) for s in sol_jit])[i_insert]
        error = sum((simulation_output - (interpolated_values-df[1][0]))**2)  # Sum of squared errors
        return error
    
    except:
       return float('inf')


def estimate_feedback():
    # set bounds
    a_f0 = a_f
    a_f_bounds = (-20e-5, -1.0e-5)

    a_b0 = -a_b
    a_b_bounds = (-5e-5, 5e-5)

    a_c0 = -a_c
    a_c_bounds = (0.5e-5, 20e-5)

    initial_guess = [a_f0,a_b0,a_c0]
    bounds = [a_f_bounds,a_b_bounds,a_c_bounds]

    # minimize
    result = minimize(sumSq_initial, initial_guess, bounds=bounds,method='Nelder-Mead')
    return result

def sumSq_initial(params):
    try:
        # run
        sol_jit = relax_feedback(params)

        # calculate error
        simulation_output = np.array([P*(s[6]-sol_jit[i_insert[0]-1][6]) for s in sol_jit])[i_insert]
        error = sum((simulation_output - (interpolated_values-P))**2)  # Sum of squared errors
        return error
    
    except:
       return float('inf')

def estimate_feedback():
    # set bounds
    a_f0 = a_f
    a_f_bounds = (-20e-5, -1.0e-5)

    a_b0 = -a_b
    a_b_bounds = (-5e-5, 5e-5)

    a_c0 = -a_c
    a_c_bounds = (0.5e-5, 20e-5)

    initial_guess = [a_f0,a_b0,a_c0]
    bounds = [a_f_bounds,a_b_bounds,a_c_bounds]

    # minimize
    result = minimize(sumSq_dP, 
                      initial_guess, 
                      bounds=bounds,
                      method='Nelder-Mead',
                      options = {'adaptive': True}
                      )

    # result = initial_guess

    return result

def main():
    # 
    result = estimate_feedback()

    # 
    print(result.x)
    # Assuming 'result.x' contains the value you want to write to the file
    value_to_write = result.x

    # Specify the file path where you want to save the value
    file_path = "estimate_hA_feedback.txt"

    # Open the file in write mode and write the value to it
    with open(file_path, "w") as file:
        file.write(str(value_to_write))

    # The value has been written to the file
    print(f"Value has been written to {file_path}")

    # sol = relax_feedback(result)
    # plt.plot([P*s[6] for s in sol])
    # plt.savefig('P_out.png')


if __name__ == "__main__":
    main()