from relaxations_modular import relax_hA_analytical
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
    
FUEL_IN_ORNL = F_to_K(1209)
FUEL_OUT_ORNL = F_to_K(1522)

COOLANT_IN_ORNL = F_to_K(1226)
COOLANT_OUT_ORNL = F_to_K(1335)

DT_FUEL_ORNL = F_to_K(1522)-F_to_K(1209)
DT_COOLANT_ORNL = F_to_K(1335)-F_to_K(1226)


def sumSq_hA(params):

    _, sol_jit = relax_hA_analytical(params) 

    fuel_in = (sol_jit[-1][15] + sol_jit[-1][20])/2
    fuel_out = sol_jit[-1][1]
    coolant_in = (sol_jit[-1][25]+sol_jit[-1][30])/2
    coolant_out = sol_jit[-1][4]

    fuel_in_error = ((fuel_in - FUEL_IN_ORNL)/FUEL_IN_ORNL)**2
    fuel_out_error = ((fuel_out - FUEL_OUT_ORNL)/FUEL_OUT_ORNL)**2
    coolant_in_error = ((coolant_in-COOLANT_IN_ORNL)/coolant_in)**2
    coolant_out_error = ((coolant_out-COOLANT_OUT_ORNL)/coolant_out)**2

    tot = fuel_in_error + fuel_out_error + coolant_in_error + coolant_out_error

    return tot

def estimate_hA():

    range_factor = 20

    ft_c = hA_ft_c
    ft_c_lims = (ft_c/range_factor,range_factor*ft_c)

    tc_c = hA_tc_c
    tc_c_lims = (tc_c/range_factor,range_factor*tc_c)

    mc_c = hA_mc_c
    mc_c_lims = (mc_c/range_factor,range_factor*mc_c)

    ft_hx = hA_ft_hx
    ft_hx_lims = (ft_hx/range_factor,range_factor*ft_hx)

    ht_hx = hA_ht_hx
    ht_hx_lims = (ht_hx/range_factor,range_factor*ht_hx)

    ct_hx = hA_ct_hx
    ct_hx_lims = (ct_hx/range_factor,range_factor*ct_hx)

    th_hxch = hA_th_hxch
    th_hxch_lims = (th_hxch/range_factor,range_factor*th_hxch)

    ht_hxhw = hA_ht_hxhw
    ht_hxhw_lims = (ht_hxhw/range_factor,range_factor*ht_hxhw)

    tw_hxhw = hA_tw_hxhw
    tw_hxhw_lims = (tw_hxhw/range_factor,range_factor*tw_hxhw)

    ht_hxhwc = hA_ht_hxhwc
    ht_hxhwc_lims = (ht_hxhwc/range_factor,range_factor*ht_hxhwc)

    tw_hxhwc = hA_tw_hxhwc
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
        sumSq_hA, 
        initial_guess, 
        bounds=bounds,
        method='Nelder-Mead',
        options = {'adaptive': True}
    )

    return result

def main():
    # # %%
    # result = estimate_hA()

    # # # %%
    # print(result.x)
    # # Assuming 'result.x' contains the value you want to write to the file
    # value_to_write = result.x

    # # Specify the file path where you want to save the value
    # file_path = "estimate_hA_ss_modular.txt"

    # # Open the file in write mode and write the value to it
    # with open(file_path, "w") as file:
    #     file.write(str(value_to_write))

    # # The value has been written to the file
    # print(f"Value has been written to {file_path}")

    params = [0.0169877,  0.00501597, 0.00060885, 0.00862134, 0.00350632, 
          0.01010759, 0.00213548, 0.00570137, 0.07343354, 0.00174456, 
          0.01248193]
    m, sol_jit = relax_hA_analytical(params)

    for idx, e in enumerate(m.dydt):
        # print(f"{m.get_node_by_index(idx).name}: {e}")
        print(f"{m.get_node_by_index(idx).name}: {m.get_node_by_index(idx).y0}")

if __name__ == '__main__':
    main()