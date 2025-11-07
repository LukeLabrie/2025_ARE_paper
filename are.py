from parameters import are_parameters
from msrDynamics import Node, System 
from jitcdde import y, t
import numpy as np

def build_steady_state_model(p: dict,
                             mann = False):

    # define system without delays, and n = 1.0
    
    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = p['m_f_c']/2, scp = p['scp_f'], W = p['W_f'], y0 = p['T0_c_f1'], name = "c_f1")
    c_f2 = Node(m = p['m_f_c']/2, scp = p['scp_f'], W = p['W_f'], y0 = p['T0_c_f2'], name = "c_f2")
    c_t1 = Node(m = p['m_t'], scp = p['scp_t'], y0 = p['T0_c_t1'], name = "c_t1")
    c_c1 = Node(m = p['m_c_c']/2, scp = p['scp_c'], W = p['W_c'], y0 = p['T0_c_c1'], name = "c_c1")
    c_c2 = Node(m = p['m_c_c']/2, scp = p['scp_c'], W = p['W_c'], y0 = p['T0_c_c2'], name = "c_c2") 
    c_m1 = Node(m = p['m_m_c'], scp = p['scp_m'], y0 = p['T0_c_m'], name = "c_m1")
    n = Node(y0 = p['n_frac0'], name = "n")
    C1 = Node(y0 = p['C0'][0], name = "C1")
    C2 = Node(y0 = p['C0'][1], name = "C2")
    C3 = Node(y0 = p['C0'][2], name = "C3")
    C4 = Node(y0 = p['C0'][3], name = "C4")
    C5 = Node(y0 = p['C0'][4], name = "C5")
    C6 = Node(y0 = p['C0'][5], name = "C6")
    rho = Node(y0 = p['rho_0'], name = "rho")

    # FUEL-HELIUM HX1
    hx_fh1_f2 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f2'], name = "hx_fh1_f2")
    hx_fh1_f1 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f1'], name = "hx_fh1_f1")
    hx_fh1_t1 = Node(m = p['m_t_hxfh'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_fh1_t1")
    hx_fh1_h1 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h1'], name = "hx_fh1_h1")
    hx_fh1_h2 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h2'], name = "hx_fh1_h2")

    # FUEL-HELIUM HX2
    hx_fh2_f2 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f2'], name = "hx_fh2_f2")
    hx_fh2_f1 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f1'], name = "hx_fh2_f1")
    hx_fh2_t1 = Node(m = p['m_t_hxfh'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_fh2_t1")
    hx_fh2_h1 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h1'], name = "hx_fh2_h1")
    hx_fh2_h2 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h2'], name = "hx_fh2_h2")

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c1'], name = "hx_ch1_c1")
    hx_ch1_c2 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c2'], name = "hx_ch1_c2")
    hx_ch1_t1 = Node(m = p['m_t_hxch'],   scp = p['scp_t'],                  y0 = p['T0_hch_t1'], name = "hx_ch1_t1")
    hx_ch1_h1 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h1'], name = "hx_ch1_h1")
    hx_ch1_h2 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h2'], name = "hx_ch1_h2")

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c1'], name = "hx_ch2_c1")
    hx_ch2_c2 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c2'], name = "hx_ch2_c2")
    hx_ch2_t1 = Node(m = p['m_t_hxch'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_ch2_t1")
    hx_ch2_h1 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h1'], name = "hx_ch2_h1")
    hx_ch2_h2 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h2'], name = "hx_ch2_h2")

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h1'], name = "hx_hwf1_h1")
    hx_hwf1_h2 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h2'], name = "hx_hwf1_h2")
    hx_hwf1_t1 = Node(m = p['m_t_hxhwf'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwf1_t1")
    hx_hwf1_w1 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w1'], name = "hx_hwf1_w1")
    hx_hwf1_w2 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w2'], name = "hx_hwf1_w2")

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h1'], name = "hx_hwf2_h1")
    hx_hwf2_h2 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h2'], name = "hx_hwf2_h2")
    hx_hwf2_t1 = Node(m = p['m_t_hxhwf'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwf2_t1")
    hx_hwf2_w1 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w1'], name = "hx_hwf2_w1")
    hx_hwf2_w2 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w2'], name = "hx_hwf2_w2")

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h1'], name = "hx_hwc1_h1")
    hx_hwc1_h2 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h2'], name = "hx_hwc1_h2")
    hx_hwc1_t1 = Node(m = p['m_t_hxhwc'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwc1_t1")
    hx_hwc1_w1 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w1'], name = "hx_hwc1_w1")
    hx_hwc1_w2 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w2'], name = "hx_hwc1_w2")

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h1'], name = "hx_hwc2_h1")
    hx_hwc2_h2 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h2'], name = "hx_hwc2_h2")
    hx_hwc2_t1 = Node(m = p['m_t_hxhwc'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwc2_t1")
    hx_hwc2_w1 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w1'], name = "hx_hwc2_w1")
    hx_hwc2_w2 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w2'], name = "hx_hwc2_w2")

    ARE.add_nodes([c_f1,c_f2,c_t1,c_c1,c_c2,c_m1,n,C1,C2,C3,C4,C5,C6,rho,
            hx_fh1_f1,hx_fh1_f2,hx_fh1_t1,hx_fh1_h1,hx_fh1_h2,
            hx_fh2_f1,hx_fh2_f2,hx_fh2_t1,hx_fh2_h1,hx_fh2_h2,
            hx_ch1_c1,hx_ch1_c2,hx_ch1_t1,hx_ch1_h1,hx_ch1_h2,
            hx_ch2_c1,hx_ch2_c2,hx_ch2_t1,hx_ch2_h1,hx_ch2_h2,
            hx_hwf1_h1,hx_hwf1_h2,hx_hwf1_t1,hx_hwf1_w1,hx_hwf1_w2,
            hx_hwf2_h1,hx_hwf2_h2,hx_hwf2_t1,hx_hwf2_w1,hx_hwf2_w2,
            hx_hwc1_h1,hx_hwc1_h2,hx_hwc1_t1,hx_hwc1_w1,hx_hwc1_w2,
            hx_hwc2_h1,hx_hwc2_h2,hx_hwc2_t1,hx_hwc2_w1,hx_hwc2_w2,
            ])
    
    

    # CORE
    # c_f1.set_dTdt_advective(source = (hx_fh1_f2.y(t-tau_hx_c_f)+hx_fh2_f2.y(t-tau_hx_c_f))/2)
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y()+hx_fh2_f2.y())/2) 
    c_f1.set_dTdt_internal(source = [1.0], k = [p['k_f1']*p['P']])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [p['hA_ft_c']/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [1.0], k = [p['k_f2']*p['P']])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [p['hA_ft_c']/2,p['hA_ft_c']/2,p['hA_tc_c']/2,p['hA_tc_c']/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [p['hA_ft_c']/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [p['hA_ft_c']/2,p['hA_ft_c']/2,p['hA_tc_c']/2,p['hA_tc_c']/2])
    c_t1.set_dTdt_internal(source = [1.0], k = [p['k_inc']*p['P']])

    # c_c1.set_dTdt_advective(source = (hx_ch1_c2.y(t-tau_c_hx_f)+hx_ch2_c2.y(t-tau_c_hx_f))/2)
    c_c1.set_dTdt_internal(source = [1.0], k = [p['k_c1']*p['P']])
    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y()+hx_ch2_c2.y())/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [p['hA_tc_c']/2,p['hA_mc_c']/2])

    c_c2.set_dTdt_internal(source = [1.0], k = [p['k_c2']*p['P']])
    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [p['hA_tc_c']/2,p['hA_mc_c']/2])

    c_m1.set_dTdt_internal(source = [1.0], k = [p['k_m']*p['P']])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [p['hA_mc_c']/2]*2)

    n.set_dndt(rho.y(), p['beta_t'], p['Lam'], p['lam'], [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    C1.set_dcdt(n = 1.0, beta = p['beta'][0], Lambda = p['Lam'], lam = p['lam'][0], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    C2.set_dcdt(n = 1.0, beta = p['beta'][1], Lambda = p['Lam'], lam = p['lam'][1], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    C3.set_dcdt(n = 1.0, beta = p['beta'][2], Lambda = p['Lam'], lam = p['lam'][2], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    C4.set_dcdt(n = 1.0, beta = p['beta'][3], Lambda = p['Lam'], lam = p['lam'][3], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    C5.set_dcdt(n = 1.0, beta = p['beta'][4], Lambda = p['Lam'], lam = p['lam'][4], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    C6.set_dcdt(n =1.0,  beta = p['beta'][5], Lambda = p['Lam'], lam = p['lam'][5], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = True)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[p['a_f']/2,p['a_f']/2,p['a_b'],p['a_c']/2,p['a_c']/2])

    # FUEL-HELIUM HX1
    # hx_fh1_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y())
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ft_hx']/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ft_hx']/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])

    # hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y(t-tau_h))
    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y())
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ht_hx']/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ht_hx']/2])

    # FUEL-HELIUM HX2
    # hx_fh2_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y())
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ft_hx']/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ft_hx']/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ht_hx']/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ht_hx']/2])

    # COOLANT-HELIUM HX1
    # hx_ch1_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y())
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_ct_hx']/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_ct_hx']/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])


    # hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y(t-tau_h))
    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y())
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_th_hxch']/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_th_hxch']/2])

    # COOLANT-HELIUM HX2
    # hx_ch2_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y())
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_ct_hx']/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_ct_hx']/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])


    # hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y(t-tau_h))
    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y())
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_th_hxch']/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_th_hxch']/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_ht_hxhw']/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_ht_hxhw']/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])


    hx_hwf1_w1.set_dTdt_advective(source = p['T0_hhwf_w1'])
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_tw_hxhw']/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_tw_hxhw']/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_ht_hxhw']/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_ht_hxhw']/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])

    hx_hwf2_w1.set_dTdt_advective(source = p['T0_hhwf_w1'])
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_tw_hxhw']/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_tw_hxhw']/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_ht_hxhwc']/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_ht_hxhwc']/2])


    hx_hwc1_w1.set_dTdt_advective(source = p['T0_hhwc_w1'])
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_ht_hxhwc']/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_ht_hxhwc']/2])


    hx_hwc2_w1.set_dTdt_advective(source = p['T0_hhwc_w1'])
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    return ARE

def build_model(p,
                mann:bool = False,
                t_insertion: float = None,
                times: np.ndarray = None,
                ):
    
    # constructs full dynamic model of the ARE system

    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = p['m_f_c']/2, scp = p['scp_f'], W = p['W_f'], y0 = p['T0_c_f1'], name = "c_f1")
    c_f2 = Node(m = p['m_f_c']/2, scp = p['scp_f'], W = p['W_f'], y0 = p['T0_c_f2'], name = "c_f2")
    c_t1 = Node(m = p['m_t'], scp = p['scp_t'], y0 = p['T0_c_t1'], name = "c_t1")
    c_c1 = Node(m = p['m_c_c']/2, scp = p['scp_c'], W = p['W_c'], y0 = p['T0_c_c1'], name = "c_c1")
    c_c2 = Node(m = p['m_c_c']/2, scp = p['scp_c'], W = p['W_c'], y0 = p['T0_c_c2'], name = "c_c2") 
    c_m1 = Node(m = p['m_m_c'], scp = p['scp_m'], y0 = p['T0_c_m'], name = "c_m1")
    n = Node(y0 = p['n_frac0'], name = "n")
    C1 = Node(y0 = p['C0'][0], name = "C1")
    C2 = Node(y0 = p['C0'][1], name = "C2")
    C3 = Node(y0 = p['C0'][2], name = "C3")
    C4 = Node(y0 = p['C0'][3], name = "C4")
    C5 = Node(y0 = p['C0'][4], name = "C5")
    C6 = Node(y0 = p['C0'][5], name = "C6")
    rho = Node(y0 = p['rho_0'], name = "rho")

    if t_insertion is not None:
        # add reactivity input
        inserted = 400e-5
        def rho_insert(t):
            if (t<t_insertion):
                return 0.0
            elif (t<(t_insertion+p['insert_duration'])):
                return ((t-t_insertion))*(inserted/p['insert_duration']) # linear
            elif (t < p['t_wd']):
                return inserted
            elif (t < p['t_wd']+p['insert_duration']):
                return inserted-((t-p['t_wd']))*(inserted/p['insert_duration']) # linear
            else:
                return 0.0

        rho_ext = ARE.add_input(rho_insert, times)
    else:
        rho_ext = 0.0

    # FUEL-HELIUM HX1
    hx_fh1_f2 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f2'], name = "hx_fh1_f2")
    hx_fh1_f1 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f1'], name = "hx_fh1_f1")
    hx_fh1_t1 = Node(m = p['m_t_hxfh'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_fh1_t1")
    hx_fh1_h1 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h1'], name = "hx_fh1_h1")
    hx_fh1_h2 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h2'], name = "hx_fh1_h2")

    # FUEL-HELIUM HX2
    hx_fh2_f2 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f2'], name = "hx_fh2_f2")
    hx_fh2_f1 = Node(m = p['m_f_hx']/2,   scp = p['scp_f'], W = p['W_f']/2,  y0 = p['T0_hfh_f1'], name = "hx_fh2_f1")
    hx_fh2_t1 = Node(m = p['m_t_hxfh'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_fh2_t1")
    hx_fh2_h1 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h1'], name = "hx_fh2_h1")
    hx_fh2_h2 = Node(m = p['m_h_hxfh']/2, scp = p['scp_h'], W = p['W_h_fh'], y0 = p['T0_hfh_h2'], name = "hx_fh2_h2")

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c1'], name = "hx_ch1_c1")
    hx_ch1_c2 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c2'], name = "hx_ch1_c2")
    hx_ch1_t1 = Node(m = p['m_t_hxch'],   scp = p['scp_t'],                  y0 = p['T0_hch_t1'], name = "hx_ch1_t1")
    hx_ch1_h1 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h1'], name = "hx_ch1_h1")
    hx_ch1_h2 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h2'], name = "hx_ch1_h2")

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c1'], name = "hx_ch2_c1")
    hx_ch2_c2 = Node(m = p['m_c_hx']/2,   scp = p['scp_c'], W = p['W_c']/2,  y0 = p['T0_hch_c2'], name = "hx_ch2_c2")
    hx_ch2_t1 = Node(m = p['m_t_hxch'],   scp = p['scp_t'],                  y0 = p['T0_hfh_t1'], name = "hx_ch2_t1")
    hx_ch2_h1 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h1'], name = "hx_ch2_h1")
    hx_ch2_h2 = Node(m = p['m_h_hxch']/2, scp = p['scp_h'], W = p['W_h_ch'], y0 = p['T0_hch_h2'], name = "hx_ch2_h2")

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h1'], name = "hx_hwf1_h1")
    hx_hwf1_h2 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h2'], name = "hx_hwf1_h2")
    hx_hwf1_t1 = Node(m = p['m_t_hxhwf'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwf1_t1")
    hx_hwf1_w1 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w1'], name = "hx_hwf1_w1")
    hx_hwf1_w2 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w2'], name = "hx_hwf1_w2")

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h1'], name = "hx_hwf2_h1")
    hx_hwf2_h2 = Node(m = p['m_h_hxhwf']/2, scp = p['scp_h'], W = p['W_h_fh'],   y0 = p['T0_hhwf_h2'], name = "hx_hwf2_h2")
    hx_hwf2_t1 = Node(m = p['m_t_hxhwf'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwf2_t1")
    hx_hwf2_w1 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w1'], name = "hx_hwf2_w1")
    hx_hwf2_w2 = Node(m = p['m_w_hxhwf']/2, scp = p['scp_w'], W = p['W_hhwf_w'], y0 = p['T0_hhwf_w2'], name = "hx_hwf2_w2")

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h1'], name = "hx_hwc1_h1")
    hx_hwc1_h2 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h2'], name = "hx_hwc1_h2")
    hx_hwc1_t1 = Node(m = p['m_t_hxhwc'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwc1_t1")
    hx_hwc1_w1 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w1'], name = "hx_hwc1_w1")
    hx_hwc1_w2 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w2'], name = "hx_hwc1_w2")

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h1'], name = "hx_hwc2_h1")
    hx_hwc2_h2 = Node(m = p['m_h_hxhwc']/2, scp = p['scp_h'], W = p['W_h_ch'],   y0 = p['T0_hhwc_h2'], name = "hx_hwc2_h2")
    hx_hwc2_t1 = Node(m = p['m_t_hxhwc'],   scp = p['scp_t'],                    y0 = p['T0_hhwf_t1'], name = "hx_hwc2_t1")
    hx_hwc2_w1 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w1'], name = "hx_hwc2_w1")
    hx_hwc2_w2 = Node(m = p['m_w_hxhwc']/2, scp = p['scp_w'], W = p['W_hhwc_w'], y0 = p['T0_hhwc_w2'], name = "hx_hwc2_w2")

    ARE.add_nodes([c_f1,c_f2,c_t1,c_c1,c_c2,c_m1,n,C1,C2,C3,C4,C5,C6,rho,
            hx_fh1_f1,hx_fh1_f2,hx_fh1_t1,hx_fh1_h1,hx_fh1_h2,
            hx_fh2_f1,hx_fh2_f2,hx_fh2_t1,hx_fh2_h1,hx_fh2_h2,
            hx_ch1_c1,hx_ch1_c2,hx_ch1_t1,hx_ch1_h1,hx_ch1_h2,
            hx_ch2_c1,hx_ch2_c2,hx_ch2_t1,hx_ch2_h1,hx_ch2_h2,
            hx_hwf1_h1,hx_hwf1_h2,hx_hwf1_t1,hx_hwf1_w1,hx_hwf1_w2,
            hx_hwf2_h1,hx_hwf2_h2,hx_hwf2_t1,hx_hwf2_w1,hx_hwf2_w2,
            hx_hwc1_h1,hx_hwc1_h2,hx_hwc1_t1,hx_hwc1_w1,hx_hwc1_w2,
            hx_hwc2_h1,hx_hwc2_h2,hx_hwc2_t1,hx_hwc2_w1,hx_hwc2_w2,
            ])
    
    

    # CORE
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y(t-p['tau_hx_c_f'])+hx_fh2_f2.y(t-p['tau_hx_c_f']))/2)
    c_f1.set_dTdt_internal(source = [n.y()], k = [p['k_f1']*p['P']])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [p['hA_ft_c']/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [n.y()], k = [p['k_f2']*p['P']])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [p['hA_ft_c']/2,p['hA_ft_c']/2,p['hA_tc_c']/2,p['hA_tc_c']/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [p['hA_ft_c']/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [p['hA_ft_c']/2,p['hA_ft_c']/2,p['hA_tc_c']/2,p['hA_tc_c']/2])
    c_t1.set_dTdt_internal(source = [n.y()], k = [p['k_inc']*p['P']])

    c_c1.set_dTdt_internal(source = [n.y()], k = [p['k_c1']*p['P']])
    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y(t-p['tau_c_hx_f'])+hx_ch2_c2.y(t-p['tau_c_hx_f']))/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [p['hA_tc_c']/2,p['hA_mc_c']/2])

    c_c2.set_dTdt_internal(source = [n.y()], k = [p['k_c2']*p['P']])
    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [p['hA_tc_c']/2,p['hA_mc_c']/2])

    c_m1.set_dTdt_internal(source = [n.y()], k = [p['k_m']*p['P']])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [p['hA_mc_c']/2]*2)

    n.set_dndt(rho.y() + rho_ext, p['beta_t'], p['Lam'], p['lam'], [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    C1.set_dcdt(n = n.y(), beta = p['beta'][0], Lambda = p['Lam'], lam = p['lam'][0], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    C2.set_dcdt(n = n.y(), beta = p['beta'][1], Lambda = p['Lam'], lam = p['lam'][1], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    C3.set_dcdt(n = n.y(), beta = p['beta'][2], Lambda = p['Lam'], lam = p['lam'][2], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    C4.set_dcdt(n = n.y(), beta = p['beta'][3], Lambda = p['Lam'], lam = p['lam'][3], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    C5.set_dcdt(n = n.y(), beta = p['beta'][4], Lambda = p['Lam'], lam = p['lam'][4], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    C6.set_dcdt(n =n.y(),  beta = p['beta'][5], Lambda = p['Lam'], lam = p['lam'][5], t_c = p['tau_c'], t_l = p['tau_l'], flow = True, force_steady_state = False)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[p['a_f']/2,p['a_f']/2,p['a_b'],p['a_c']/2,p['a_c']/2])

    # FUEL-HELIUM HX1
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y(t-p['tau_c_hx_f']))
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ft_hx']/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ft_hx']/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])

    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y(t-p['tau_h']))
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ht_hx']/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [p['hA_ht_hx']/2])

    # FUEL-HELIUM HX2
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y(t-p['tau_c_hx_f']))
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ft_hx']/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ft_hx']/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [p['hA_ft_hx']/2,p['hA_ft_hx']/2,p['hA_ht_hx']/2,p['hA_ht_hx']/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ht_hx']/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [p['hA_ht_hx']/2])

    # COOLANT-HELIUM HX1
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y(t-p['tau_c_hx_f']))
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_ct_hx']/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_ct_hx']/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])


    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y(t-p['tau_h']))
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_th_hxch']/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [p['hA_th_hxch']/2])

    # COOLANT-HELIUM HX2
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y(t-p['tau_c_hx_f']))
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_ct_hx']/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_ct_hx']/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [p['hA_ct_hx']/2,p['hA_ct_hx']/2,p['hA_th_hxch']/2,p['hA_th_hxch']/2])


    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y(t-p['tau_h']))
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_th_hxch']/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [p['hA_th_hxch']/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_ht_hxhw']/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_ht_hxhw']/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])


    hx_hwf1_w1.set_dTdt_advective(source = p['T0_hhwf_w1'])
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_tw_hxhw']/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [p['hA_tw_hxhw']/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_ht_hxhw']/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_ht_hxhw']/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [p['hA_ht_hxhw']/2,p['hA_ht_hxhw']/2,p['hA_tw_hxhw']/2,p['hA_tw_hxhw']/2])

    hx_hwf2_w1.set_dTdt_advective(source = p['T0_hhwf_w1'])
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_tw_hxhw']/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [p['hA_tw_hxhw']/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_ht_hxhwc']/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_ht_hxhwc']/2])


    hx_hwc1_w1.set_dTdt_advective(source = p['T0_hhwc_w1'])
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_ht_hxhwc']/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [p['hA_ht_hxhwc']/2,p['hA_ht_hxhwc']/2,p['hA_tw_hxhwc']/2,p['hA_tw_hxhwc']/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_ht_hxhwc']/2])


    hx_hwc2_w1.set_dTdt_advective(source = p['T0_hhwc_w1'])
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [p['hA_tw_hxhwc']/2])

    return ARE