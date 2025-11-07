from parameters import *
import numpy as np
from scipy.optimize import minimize
from are import build_steady_state_model, build_model
from symengine import Function, Symbol 
from sympy import symbols, Eq, linsolve
    
FUEL_IN_ORNL = F_to_K(1209)
FUEL_OUT_ORNL = F_to_K(1522)

COOLANT_IN_ORNL = F_to_K(1226)
COOLANT_OUT_ORNL = F_to_K(1335)

DT_FUEL_ORNL = F_to_K(1522)-F_to_K(1209)
DT_COOLANT_ORNL = F_to_K(1335)-F_to_K(1226)


def _sumSq_hA(hA_params, p):

    p['hA_ft_c'] =     hA_params[0]
    p['hA_tc_c'] =     hA_params[1]
    p['hA_mc_c'] =     hA_params[2]
    p['hA_ft_hx'] =    hA_params[3]
    p['hA_ht_hx'] =    hA_params[4]
    p['hA_ct_hx'] =    hA_params[5]
    p['hA_th_hxch'] =  hA_params[6]
    p['hA_ht_hxhw'] =  hA_params[7]
    p['hA_tw_hxhw'] =  hA_params[8]
    p['hA_ht_hxhwc'] = hA_params[9]
    p['hA_tw_hxhwc'] = hA_params[10]

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
    ARE_dynamic = build_model(p)
    y0_analytical = sol_list_temps[:6] + sol_list_precursors + sol_list_temps[6:]

    for idx, n in enumerate(ARE_dynamic.nodes):
        ARE_dynamic.nodes[n].y0 = y0_analytical[idx]

    # solve
    T = np.arange(0,30,1.0)
    sol = ARE_dynamic.solve(T,
                            max_delay = p['tau_l'], 
                            populate_nodes = False, 
                            md_step = 0.0001, 
                            abs_tol = 1.0e-12, 
                            rel_tol = 5.0e-8,)

    fuel_in = (sol[-1][15] + sol[-1][20])/2
    fuel_out = sol[-1][1]
    coolant_in = (sol[-1][25]+sol[-1][30])/2
    coolant_out = sol[-1][4]

    fuel_in_error = ((fuel_in - FUEL_IN_ORNL)/FUEL_IN_ORNL)**2
    fuel_out_error = ((fuel_out - FUEL_OUT_ORNL)/FUEL_OUT_ORNL)**2
    coolant_in_error = ((coolant_in-COOLANT_IN_ORNL)/coolant_in)**2
    coolant_out_error = ((coolant_out-COOLANT_OUT_ORNL)/coolant_out)**2

    tot = fuel_in_error + fuel_out_error + coolant_in_error + coolant_out_error
    print(f"error: {tot}")
    return tot

def _estimate_hA(p):


    range_factor = 20

    ft_c = p['hA_ft_c']
    ft_c_lims = (ft_c/range_factor,range_factor*ft_c)

    tc_c = p['hA_tc_c']
    tc_c_lims = (tc_c/range_factor,range_factor*tc_c)

    mc_c = p['hA_mc_c']
    mc_c_lims = (mc_c/range_factor,range_factor*mc_c)

    ft_hx = p['hA_ft_hx']
    ft_hx_lims = (ft_hx/range_factor,range_factor*ft_hx)

    ht_hx = p['hA_ht_hx']
    ht_hx_lims = (ht_hx/range_factor,range_factor*ht_hx)

    ct_hx = p['hA_ct_hx']
    ct_hx_lims = (ct_hx/range_factor,range_factor*ct_hx)

    th_hxch = p['hA_th_hxch']
    th_hxch_lims = (th_hxch/range_factor,range_factor*th_hxch)

    ht_hxhw = p['hA_ht_hxhw']
    ht_hxhw_lims = (ht_hxhw/range_factor,range_factor*ht_hxhw)

    tw_hxhw = p['hA_tw_hxhw']
    tw_hxhw_lims = (tw_hxhw/range_factor,range_factor*tw_hxhw)

    ht_hxhwc = p['hA_ht_hxhwc']
    ht_hxhwc_lims = (ht_hxhwc/range_factor,range_factor*ht_hxhwc)

    tw_hxhwc = p['hA_tw_hxhwc']
    tw_hxhwc_lims = (tw_hxhwc/range_factor,range_factor*tw_hxhwc)

    # T0_m = T0_c_m
    # T0_m_lims = (T0_c_m-600,T0_c_m+600)


    initial_guess = [ft_c,tc_c,mc_c,ft_hx,ht_hx,ct_hx,th_hxch,ht_hxhw,
                     tw_hxhw,ht_hxhwc,tw_hxhwc] # ,T0_m]
    
    bounds = [ft_c_lims,tc_c_lims,mc_c_lims,ft_hx_lims,ht_hx_lims,ct_hx_lims,
              th_hxch_lims,ht_hxhw_lims,tw_hxhw_lims,ht_hxhwc_lims,tw_hxhwc_lims,]
              # T0_m_lims]

    # relax_hA(initial_guess)

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
    param_set = 'baseline'
    params = are_parameters(param_set)

    # find solution
    res = _estimate_hA(params)
    print(res.x)

    # value to write to file
    file_path = f"estimate_hA_{param_set}.txt"

    with open(file_path, "w") as file:
        file.write(str(res.x))

    print(f"Value has been written to {file_path}")


if __name__ == '__main__':
    main()