# %%
from relaxations_with_feedbacks import relax_hA
from parameters_updated_heating import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
    
DT_FUEL_ORNL = F_to_K(1522)-F_to_K(1209)
DT_COOLANT_ORNL = F_to_K(1335)-F_to_K(1226)

def sumSq_hA(params):

    sol_jit = relax_hA(params) 

    # calculate error
    pow = sol_jit[-1][6]
    pow_error = ((pow - 2.10)/2.10)**2  

    fuel_in = (sol_jit[-1][15] + sol_jit[-1][20])/2
    fuel_out = sol_jit[-1][1]
    dT_fuel = fuel_out-fuel_in
    dT_fuel_error = ((dT_fuel - DT_FUEL_ORNL)/DT_FUEL_ORNL)**2

    # fuel_in_error = ((fuel_in-F_to_K(1209))/fuel_in)**2
    # fuel_out_error = ((fuel_out-F_to_K(1522))/fuel_out)**2

    coolant_in = (sol_jit[-1][25]+sol_jit[-1][30])/2
    coolant_out = sol_jit[-1][4]
    dT_coolant = coolant_out-coolant_in
    dT_coolant_error = ((dT_coolant - DT_COOLANT_ORNL)/DT_COOLANT_ORNL)**2

    #coolant_in_error = ((coolant_in-F_to_K(1226))/coolant_in)**2
    #coolant_out_error = ((coolant_out-F_to_K(1335))/coolant_out)**2
    

    tot = pow_error + dT_fuel_error + dT_coolant_error

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

    fb_f = a_f
    fb_f_lims = (-10e-5, -0.1e-5)

    fb_b = a_b
    fb_b_lims = (-2e-5,2e-5)

    fb_c = a_c
    fb_c_lims = (-5e-5,5e-5)

    initial_guess = [ft_c,tc_c,mc_c,ft_hx,ht_hx,ct_hx,th_hxch,ht_hxhw,
                     tw_hxhw,ht_hxhwc,tw_hxhwc, fb_f, fb_b, fb_c] 
    
    # ,T0_m]
    
    bounds = [ft_c_lims,tc_c_lims,mc_c_lims,ft_hx_lims,ht_hx_lims,ct_hx_lims,
              th_hxch_lims,ht_hxhw_lims,tw_hxhw_lims,ht_hxhwc_lims,tw_hxhwc_lims,
              fb_f_lims, fb_b_lims, fb_c_lims]
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
    result = estimate_hA()

    # # %%
    print(result.x)
    # Assuming 'result.x' contains the value you want to write to the file
    value_to_write = result.x

    # Specify the file path where you want to save the value
    file_path = "estimate_hA_ss_updated_heating.txt"

    # Open the file in write mode and write the value to it
    with open(file_path, "w") as file:
        file.write(str(value_to_write))

    # The value has been written to the file
    print(f"Value has been written to {file_path}")

    # data = np.array(result.x)
    # print(data[0])
    # print(data[-1])

    # ft_c = hA_ft_c
    # tc_c = hA_tc_c
    # mc_c = hA_mc_c
    # ft_hx = hA_ft_hx
    # ht_hx = hA_ht_hx
    # ct_hx = hA_ct_hx
    # th_hxch = hA_th_hxch
    # ht_hxhw = hA_ht_hxhw
    # tw_hxhw = hA_tw_hxhw
    # ht_hxhwc = hA_ht_hxhwc
    # tw_hxhwc = hA_tw_hxhwc

    # params = [2.26103880e-02, 6.35249293e-03, 5.01754655e-05, 7.79148981e-03,
    #       4.50255426e-03, 1.05622729e-02, 1.26345367e-03, 6.71738834e-03,
    #       1.02512617e-01, 2.30353891e-03, 7.33258046e-03]

    # # initial_guess = [ft_c,tc_c,mc_c,ft_hx,ht_hx,ct_hx,th_hxch,ht_hxhw,
    # #                  tw_hxhw,ht_hxhwc,tw_hxhwc]
    
    
    # sol = relax_hA(params)
    # # print(sol[-1][6])
    # plt.plot([P*s[6] for s in sol])
    # plt.savefig('P_out.png')

if __name__ == '__main__':
    main()