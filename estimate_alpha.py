from parameters import *
import numpy as np
from scipy.optimize import minimize
from are import build_steady_state_model, build_model
from symengine import Function, Symbol 
from sympy import symbols, Eq, linsolve
import pandas as pd
from scipy.interpolate import CubicSpline
    
# some global params
params = are_parameters('leastsq', 'baseline', replace = [('P', 2.34)])
T = np.arange(0, 1000.0, 0.1)

# read experimental data
df_reversed = pd.read_csv("../data/insertion.csv",header=None)
df = df_reversed.iloc[::-1]
df = df.reset_index(drop=True)

# get indicies for comparison
t_before_data = (1110-df[0][0])*60
duration_data = (df.iloc[-1][0]-df[0][0])*60
t_end_data = df.iloc[-1][0]
t_before_sim = params['t_ins']-t_before_data
T_insert = [t for t in T if (t > (t_before_sim)) and (t < (t_before_sim)+(duration_data))]
i_insert = [t[0] for t in enumerate(T) if (t[1] > (t_before_sim)) and (t[1] < (t_before_sim)+(duration_data))]

adj = (df[0][0])*60-T_insert[0]
df[0] = [(t*60)-adj for t in df[0]]

# adjust to reported power
d = df[1][0]-params['P']
df[1] = [p-d for p in df[1]]

# Set up interpolation
# Assuming df[0] is time and df[1] is the data you want to interpolate
spline = CubicSpline(df[0], df[1])  # Multiplying df[0] by 60 if it's in minutes

# Use the spline to interpolate at the desired times
interpolated_values = spline(T[i_insert])


def _sumSq_hA(hA_params, p):

    p['a_f'] =     hA_params[0]
    p['a_b'] =     hA_params[1]
    p['a_c'] =     hA_params[2]

    # repeat the model building process from the steady_state_notebook
    ARE_ss = build_steady_state_model(p)
    subs = {Function("current_y")(i): Symbol(f"y{i}") for i in range(len(ARE_ss.nodes))}
    exprs_subbed = [expr.subs(subs) for expr in ARE_ss.dydt]
    precursor_system = exprs_subbed[7:13]
    temp_system = exprs_subbed[:6] + exprs_subbed[14:]
    # solve each system (temps and precursors)
    ys = symbols(f'y0:{54}')
    ys_precursors = ys[7:13]
    ys_temp = ys[:6] + ys[14:]

    eqns_precursors = [Eq(expr, 0) for expr in precursor_system]
    eqns_temps = [Eq(expr, 0) for expr in temp_system]

    sol_precursors = linsolve(eqns_precursors, ys_precursors)
    sol_temps = linsolve(eqns_temps, ys_temp)

    # unpack
    sol_tuple_precursors = next(iter(sol_precursors))
    sol_tuple_temps = next(iter(sol_temps))
    sol_list_precursors = list(sol_tuple_precursors)
    sol_list_temps = list(sol_tuple_temps)

    # add the fixed variables to the precursor solution
    sol_list_precursors = [p['n_frac0']] + sol_list_precursors + [p['rho_0']]

    # build model with computed steady state condition
    ARE_dynamic = build_model(p, t_insertion = p['t_ins'], times = T)
    y0_analytical = sol_list_temps[:6] + sol_list_precursors + sol_list_temps[6:]

    for idx, n in enumerate(ARE_dynamic.nodes):
        ARE_dynamic.nodes[n].y0 = y0_analytical[idx]

    # solve
    sol = ARE_dynamic.solve(T,
                            max_delay = p['tau_l'], 
                            populate_nodes = False, 
                            md_step = 0.0001, 
                            abs_tol = 1.0e-12, 
                            rel_tol = 5.0e-8,)

    power_out = p['P']*sol[:,6]
    power_out_exp = power_out[i_insert]
    error = power_out_exp - interpolated_values
    sum_sq_error = (np.sum(error))**2

    print(f"error: {sum_sq_error}")
    return sum_sq_error

def _estimate_alpha(p):

    # set initial guess based on OpenMC estimate
    a_f0 = -6.0e-05
    a_b0 = 1.4e-05
    a_c0 = 2.3e-05

    a_f_bounds = (-20e-5, -1.0e-5)
    a_b_bounds = (-5e-5, 5e-5)
    a_c_bounds = (-5e-5, 5e-5)

    initial_guess = [a_f0,a_b0,a_c0]
    bounds = [a_f_bounds,a_b_bounds,a_c_bounds]

    result = minimize(
        _sumSq_hA, 
        initial_guess, 
        args = (p),
        bounds=bounds,
        method='Nelder-Mead',
        options = {'adaptive': True}
    )

    return result

def main():

    # get parameters
    hA_param_set = 'leastsq'
    alpha_param_set = 'baseline'
    params = are_parameters(hA_param_set, alpha_param_set,replace = [('P', 2.34)])

    # find solution
    res = _estimate_alpha(params)
    print(res.x)

    # value to write to file
    file_path = f"estimate_alpha_{alpha_param_set}.txt"

    with open(file_path, "w") as file:
        file.write(str(res.x))

    print(f"Value has been written to {file_path}")


if __name__ == '__main__':
    main()