from relaxations_modular import *
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

plt.rcParams["font.family"] = "monospace"

# Set a professional color scheme
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

tls = 10
# Function to update the style of each axis
def update_axis_style(ax, title = '', x_label='', y_label='', x_ticks=True, y_scale='linear'):
    ax.set_title(title,fontsize=tls)
    ax.set_xlabel(x_label,fontsize=tls)
    ax.set_ylabel(y_label,fontsize=tls)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', which='both', bottom=x_ticks, top=False, labelbottom=x_ticks,labelsize=tls)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True,labelsize=tls)
    ax.set_yscale(y_scale)

# read experimental data
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
interpolated_values = spline(T[i_insert])

def sumSq_dP(params):
    # run
    m, sol_jit = relax_fb_analytical(params)


    power_out = P*sol_jit[:,6]
    power_out_exp = power_out[i_insert]
    error = power_out_exp - interpolated_values
    sum_sq_error = (np.sum(error))**2

    print(f"error: {sum_sq_error}")
    plt.figure()  # Create a new figure for each plot
    update_axis_style(plt.gca(), title = f"params: {', '.join([f'{1e5*params[i]:.2f}' for i in range(len(params))])}, err: {sum_sq_error:.5f}",)
    plt.plot(T[i_insert], power_out_exp-P, label='Simulated Power')
    plt.plot(T[i_insert], interpolated_values-P, label='Experimental Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'last_run.png')  # Save with a unique filename
    plt.close()  # Close the figure to free memory

    return sum_sq_error

def estimate_feedback():
    # set bounds, initial guess based on openmc values 
    # a_f0 = -3.51e-05
    # a_b0 = -2.98e-05
    # a_c0 = 6.81e-06
    a_f0 = -6.0e-05
    a_b0 = 1.4e-05
    a_c0 = 2.3e-05

    a_f_bounds = (-20e-5, -1.0e-5)
    a_b_bounds = (-5e-5, 5e-5)
    a_c_bounds = (0.1e-5, 20e-5)

    initial_guess = [a_f0,a_b0,a_c0]
    bounds = [a_f_bounds,a_b_bounds,a_c_bounds]

    # minimize
    result = minimize(sumSq_dP, 
                      initial_guess, 
                      bounds=bounds,
                      method='Nelder-Mead',
                      options = {'adaptive': True}
                      )

    return result

def main():
    # 
    result = estimate_feedback()

    # 
    print(result.x)
    # Assuming 'result.x' contains the value you want to write to the file
    value_to_write = result.x

    # Specify the file path where you want to save the value
    file_path = "estimate_fb_analytical.txt"

    # Open the file in write mode and write the value to it
    with open(file_path, "w") as file:
        file.write(str(value_to_write))

    # The value has been written to the file
    print(f"Value has been written to {file_path}")
    # last result:
    # [-2.72670133e-05 -3.15829877e-05  7.76774068e-06]
    # m, sol = relax_fb_analytical([-4.81833507e-05,  2.03631125e-05,  2.63301128e-05])
    # power_out = P*sol[:,6]
    # power_out_exp = power_out[i_insert]
    # error = power_out_exp - interpolated_values
    # sum_sq_error = (np.sum(error))**2

    # print(f"error: {sum_sq_error}")
    # plt.figure()  # Create a new figure for each plot
    # update_axis_style(plt.gca(), title = f"params: {[-4.81833507e-05,  2.03631125e-05,  2.63301128e-05]}")
    # plt.plot(T[i_insert], power_out_exp, label='Simulated Power')
    # plt.plot(T[i_insert], interpolated_values, label='Experimental Power')
    # plt.savefig('debug.png')

    # with open("dydt_output.txt", "w") as f:
    #     for e in m.dydt:
    #         f.write(f"{str(e)}\n")
    # print(hA_ft_c)

    return None


if __name__ == "__main__":
    main()
    # params = [-4.51406644e-05, -2.11366484e-05,  2.65883301e-05]
    # params = [-3.51e-05, -2.98e-05, 6.81e-06] # openmc
    # print("openmc error", sumSq_dP(params))
    # params = [-3.52e-05, -2.98e-05,  6.67e-06] # estimated, CG
    # print("estim CG error", sumSq_dP(params))
    # params = [-4.92e-05, -1.18e-05,  4.46e-05] # estimated, powell
    # print("estim Powell error", sumSq_dP(params))
    # params = [-3.52526541e-05, -2.98101680e-05,  6.67240265e-06]
    # print("estim BFGS error", sumSq_dP(params))
    # params = [-3.54705403e-05, -2.21691953e-05,  6.33429154e-06]
    # print("estim Nelder-Mead error", sumSq_dP(params))
    # now running with different intiia