from relaxations import relax_temps
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
    
DT_FUEL_ORNL = F_to_K(1522)-F_to_K(1209)
DT_COOLANT_ORNL = F_to_K(1335)-F_to_K(1226)

def sumSq_hA(params):

    sol_jit = relax_temps(params) 

    # calculate error
    pow = sol_jit[-1][6]
    pow_error = ((pow - 2.10)/2.10)**2  

    tot = pow_error

    return tot

def estimate_hA():

    T_range = 200

    initial_guess = [T0_c_f1, T0_c_f2, T0_c_t1, T0_c_c1, T0_c_c2, T0_c_m, T0_hfh_f1, T0_hfh_f2, 
    T0_hfh_t1, T0_hfh_h1, T0_hfh_h2, T0_hch_c1, T0_hch_c2, T0_hch_t1, T0_hch_h1, 
    T0_hch_h2, T0_hhwf_h1, T0_hhwf_h2, T0_hhwf_t1, T0_hhwf_w1, T0_hhwf_w2, 
    T0_hhwc_h1, T0_hhwc_h2, T0_hhwc_t1, T0_hhwc_w1, T0_hhwc_w2]

    bounds = [(t-T_range,t+T_range) for t in initial_guess]


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
    file_path = "estimate_temps.txt"

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