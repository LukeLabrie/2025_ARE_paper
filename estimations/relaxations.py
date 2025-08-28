from jitcdde import t
from parameters import *
import matplotlib.pyplot as plt
from msrDynamics import Node, System

def relax_hA(params):

    # Unpack parameters
    hA_ft_c, hA_tc_c, hA_mc_c, hA_ft_hx, hA_ht_hx, hA_ct_hx, hA_th_hxch, \
    hA_ht_hxhw, hA_tw_hxhw, hA_ht_hxhwc, hA_tw_hxhwc = params # ,T0_c_m = params

    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f1)
    c_f2 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f2)
    c_t1 = Node(m = m_t, scp = scp_t, y0 = T0_c_t1)
    c_c1 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c1)
    c_c2 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c2) 
    c_m1 = Node(m = m_m_c, scp = scp_m, y0 = T0_c_m+75)
    n = Node(y0 = n_frac0)
    C1 = Node(y0 = C0[0])
    C2 = Node(y0 = C0[1])
    C3 = Node(y0 = C0[2])
    C4 = Node(y0 = C0[3])
    C5 = Node(y0 = C0[4])
    C6 = Node(y0 = C0[5])
    rho = Node(y0 = 0.00)

    # add reactivity input
    inserted = 0.0
    def rho_insert(t):
        if (t<t_ins):
            return 0.0
        elif (t<(t_ins+insert_duration)):
            return ((t-t_ins))*(inserted/insert_duration) # linear
        elif (t < t_wd):
            return inserted
        elif (t < t_wd+insert_duration):
            return inserted-((t-t_wd))*(inserted/insert_duration) # linear
        else:
            return 0.0

    # rho_ext = ARE.add_input(rho_insert, T)

    # FUEL-HELIUM HX1
    hx_fh1_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh1_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh1_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh1_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh1_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # FUEL-HELIUM HX2
    hx_fh2_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh2_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh2_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh2_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh2_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch1_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch1_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hch_t1)
    hx_ch1_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch1_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch2_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch2_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hfh_t1)
    hx_ch2_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch2_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf1_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf1_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf1_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf1_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf2_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf2_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf2_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf2_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc1_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc1_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwc1_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc1_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc2_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc2_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwc2_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc2_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)


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
    
    mann = False

    # CORE
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y(t-tau_hx_c_f)+hx_fh2_f2.y(t-tau_hx_c_f))/2) 
    c_f1.set_dTdt_internal(source = [n.y()], k = [k_f1*P])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [n.y()], k = [k_f2*P])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2])


    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y(t-tau_c_hx_f)+hx_ch2_c2.y(t-tau_c_hx_f))/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_m1.set_dTdt_internal(source = [n.y()], k = [k_m*P])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [hA_mc_c/2]*2)

    n.set_dndt(rho.y(), beta_t, Lam, lam, [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    C1.set_dcdt(n.y(), beta[0], Lam, lam[0], tau_c, tau_l)
    C2.set_dcdt(n.y(), beta[1], Lam, lam[1], tau_c, tau_l)
    C3.set_dcdt(n.y(), beta[2], Lam, lam[2], tau_c, tau_l)
    C4.set_dcdt(n.y(), beta[3], Lam, lam[3], tau_c, tau_l)
    C5.set_dcdt(n.y(), beta[4], Lam, lam[4], tau_c, tau_l)
    C6.set_dcdt(n.y(), beta[5], Lam, lam[5], tau_c, tau_l)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[a_f/2,a_f/2,a_b,a_c/2,a_c/2])

    # FUEL-HELIUM HX1
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])

    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y(t-tau_h))
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    # FUEL-HELIUM HX2
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    # COOLANT-HELIUM HX1
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y(t-tau_h))
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    # COOLANT-HELIUM HX2
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y(t-tau_h))
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])


    hx_hwf1_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])

    hx_hwf2_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc1_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc2_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])



    T, sol_jit = ARE.equilibrium_search(dT = 0.01, 
                                        max_delay = tau_l, 
                                        populate_nodes = False, 
                                        md_step = 0.0001, 
                                        abs_tol_eq = 1.0e-12, 
                                        rel_tol_eq = 5.0e-8,
                                        show_conv_metrics = True)
    
    return sol_jit

def relax_feedback(params):

    P = 2.2

    # Unpack parameters
    a_f, a_b, a_c = params

    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f1)
    c_f2 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f2)
    c_t1 = Node(m = m_t, scp = scp_t, y0 = T0_c_t1)
    c_c1 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c1)
    c_c2 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c2) 
    c_m1 = Node(m = m_m_c, scp = scp_m, y0 = T0_c_m+75)
    n = Node(y0 = n_frac0)
    C1 = Node(y0 = C0[0])
    C2 = Node(y0 = C0[1])
    C3 = Node(y0 = C0[2])
    C4 = Node(y0 = C0[3])
    C5 = Node(y0 = C0[4])
    C6 = Node(y0 = C0[5])
    rho = Node(y0 = 0.00)

    # add reactivity input
    inserted = 400e-5
    def rho_insert(t):
        if (t<t_ins):
            return 0.0
        elif (t<(t_ins+insert_duration)):
            return ((t-t_ins))*(inserted/insert_duration) # linear
        elif (t < t_wd):
            return inserted
        elif (t < t_wd+insert_duration):
            return inserted-((t-t_wd))*(inserted/insert_duration) # linear
        else:
            return 0.0

    rho_ext = ARE.add_input(rho_insert, T)

    # FUEL-HELIUM HX1
    hx_fh1_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh1_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh1_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh1_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh1_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # FUEL-HELIUM HX2
    hx_fh2_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh2_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh2_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh2_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh2_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch1_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch1_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hch_t1)
    hx_ch1_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch1_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch2_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch2_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hfh_t1)
    hx_ch2_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch2_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf1_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf1_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf1_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf1_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf2_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf2_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf2_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf2_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc1_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc1_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwc1_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc1_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc2_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc2_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwc2_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc2_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)


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
    
    mann = False

    # CORE
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y(t-tau_hx_c_f)+hx_fh2_f2.y(t-tau_hx_c_f))/2) 
    c_f1.set_dTdt_internal(source = [n.y()], k = [k_f1*P])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [n.y()], k = [k_f2*P])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2])


    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y(t-tau_c_hx_f)+hx_ch2_c2.y(t-tau_c_hx_f))/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_m1.set_dTdt_internal(source = [n.y()], k = [k_m*P])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [hA_mc_c/2]*2)

    n.set_dndt(rho.y() + rho_ext, beta_t, Lam, lam, [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    C1.set_dcdt(n.y(), beta[0], Lam, lam[0], tau_c, tau_l)
    C2.set_dcdt(n.y(), beta[1], Lam, lam[1], tau_c, tau_l)
    C3.set_dcdt(n.y(), beta[2], Lam, lam[2], tau_c, tau_l)
    C4.set_dcdt(n.y(), beta[3], Lam, lam[3], tau_c, tau_l)
    C5.set_dcdt(n.y(), beta[4], Lam, lam[4], tau_c, tau_l)
    C6.set_dcdt(n.y(), beta[5], Lam, lam[5], tau_c, tau_l)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[a_f/2,a_f/2,a_b,a_c/2,a_c/2])

    # FUEL-HELIUM HX1
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])

    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y(t-tau_h))
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    # FUEL-HELIUM HX2
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    # COOLANT-HELIUM HX1
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y(t-tau_h))
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    # COOLANT-HELIUM HX2
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y(t-tau_h))
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])


    hx_hwf1_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])

    hx_hwf2_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc1_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc2_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    sol_jit = ARE.solve(T, populate_nodes = False, max_delay = tau_l, md_step = 0.0001)
    
    return sol_jit

def relax_temps(params):

    # Unpack parameters
    T0_c_f1, T0_c_f2, T0_c_t1, T0_c_c1, T0_c_c2, T0_c_m, T0_hfh_f1, T0_hfh_f2, \
    T0_hfh_t1, T0_hfh_h1, T0_hfh_h2, T0_hch_c1, T0_hch_c2, T0_hch_t1, T0_hch_h1, \
    T0_hch_h2, T0_hhwf_h1, T0_hhwf_h2, T0_hhwf_t1, T0_hhwf_w1, T0_hhwf_w2, \
    T0_hhwc_h1, T0_hhwc_h2, T0_hhwc_t1, T0_hhwc_w1, T0_hhwc_w2 = params

    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f1)
    c_f2 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f2)
    c_t1 = Node(m = m_t, scp = scp_t, y0 = T0_c_t1)
    c_c1 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c1)
    c_c2 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c2) 
    c_m1 = Node(m = m_m_c, scp = scp_m, y0 = T0_c_m)
    n = Node(y0 = n_frac0)
    C1 = Node(y0 = C0[0])
    C2 = Node(y0 = C0[1])
    C3 = Node(y0 = C0[2])
    C4 = Node(y0 = C0[3])
    C5 = Node(y0 = C0[4])
    C6 = Node(y0 = C0[5])
    rho = Node(y0 = 0.00)

    # add reactivity input
    inserted = 0.0
    def rho_insert(t):
        if (t<t_ins):
            return 0.0
        elif (t<(t_ins+insert_duration)):
            return ((t-t_ins))*(inserted/insert_duration) # linear
        elif (t < t_wd):
            return inserted
        elif (t < t_wd+insert_duration):
            return inserted-((t-t_wd))*(inserted/insert_duration) # linear
        else:
            return 0.0

    # rho_ext = ARE.add_input(rho_insert, T)

    # FUEL-HELIUM HX1
    hx_fh1_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh1_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh1_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh1_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh1_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # FUEL-HELIUM HX2
    hx_fh2_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1)
    hx_fh2_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2)
    hx_fh2_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1)
    hx_fh2_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1)
    hx_fh2_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2)

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch1_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch1_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hch_t1)
    hx_ch1_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch1_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1)
    hx_ch2_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2)
    hx_ch2_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hfh_t1)
    hx_ch2_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1)
    hx_ch2_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2)

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf1_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf1_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf1_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf1_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1)
    hx_hwf2_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2)
    hx_hwf2_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1)
    hx_hwf2_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1)
    hx_hwf2_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2)

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc1_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc1_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwc_t1)
    hx_hwc1_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc1_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1)
    hx_hwc2_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2)
    hx_hwc2_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwc_t1)
    hx_hwc2_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1)
    hx_hwc2_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2)


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
    
    mann = False

    # CORE
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y(t-tau_hx_c_f)+hx_fh2_f2.y(t-tau_hx_c_f))/2) 
    c_f1.set_dTdt_internal(source = [n.y()], k = [k_f1*P])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [n.y()], k = [k_f2*P])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2])


    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y(t-tau_c_hx_f)+hx_ch2_c2.y(t-tau_c_hx_f))/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_m1.set_dTdt_internal(source = [n.y()], k = [k_m*P])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [hA_mc_c/2]*2)

    n.set_dndt(rho.y(), beta_t, Lam, lam, [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    C1.set_dcdt(n.y(), beta[0], Lam, lam[0], tau_c, tau_l)
    C2.set_dcdt(n.y(), beta[1], Lam, lam[1], tau_c, tau_l)
    C3.set_dcdt(n.y(), beta[2], Lam, lam[2], tau_c, tau_l)
    C4.set_dcdt(n.y(), beta[3], Lam, lam[3], tau_c, tau_l)
    C5.set_dcdt(n.y(), beta[4], Lam, lam[4], tau_c, tau_l)
    C6.set_dcdt(n.y(), beta[5], Lam, lam[5], tau_c, tau_l)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[a_f/2,a_f/2,a_b,a_c/2,a_c/2])

    # FUEL-HELIUM HX1
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])

    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y(t-tau_h))
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    # FUEL-HELIUM HX2
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y(t-tau_c_hx_f))
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    # COOLANT-HELIUM HX1
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y(t-tau_h))
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    # COOLANT-HELIUM HX2
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y(t-tau_c_hx_f))
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y(t-tau_h))
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])


    hx_hwf1_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])

    hx_hwf2_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc1_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc2_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])



    T, sol_jit = ARE.equilibrium_search(dT = 0.01, 
                                        max_delay = tau_l, 
                                        populate_nodes = False, 
                                        md_step = 0.0001, 
                                        abs_tol_eq = 1.0e-12, 
                                        rel_tol_eq = 5.0e-8,
                                        show_conv_metrics = True)
    
    return sol_jit

def relax_hA_analytical(params):

    sol_list_temps = [1322.42515624039,
                        1354.48864488803,
                        1325.18784857967,
                        1266.48597304708,
                        1294.95745550934,
                        1394.66207830863,
                        1257.68602233574,
                        1200.63163859416,
                        1118.73611815027,
                        562.253545888706,
                        682.789461735282,
                        1257.68602233574,
                        1200.63163859416,
                        1118.73611815028,
                        562.253545888705,
                        682.789461735284,
                        1240.37390045552,
                        1214.41791101737,
                        1190.88418143804,
                        837.055700693272,
                        1017.88947489197,
                        1240.37390045552,
                        1214.41791101738,
                        1190.88418143804,
                        837.055700693265,
                        1017.88947489197,
                        502.466512928666,
                        408.390371316368,
                        305.774161991459,
                        297.538257494245,
                        301.666494463592,
                        502.466512928666,
                        408.390371316367,
                        305.774161991460,
                        297.538257494245,
                        301.666494463592,
                        640.242769316055,
                        467.193865787768,
                        320.829033357303,
                        300.679968271512,
                        307.968357235978,
                        640.242769316054,
                        467.193865787768,
                        320.829033357303,
                        300.679968271512,
                        307.968357235978]

    sol_list_precursors = [n_frac0,
                        10.0678102111227,
                        35.3044299594146,
                        16.7728029831596,
                        18.4699598500866,
                        1.80016206438269,
                        0.217360950669434,
                        rho_0]

    sol_list_eq = sol_list_temps[:6] + sol_list_precursors + sol_list_temps[6:]

    # Unpack parameters
    hA_ft_c, hA_tc_c, hA_mc_c, hA_ft_hx, hA_ht_hx, hA_ct_hx, hA_th_hxch, \
    hA_ht_hxhw, hA_tw_hxhw, hA_ht_hxhwc, hA_tw_hxhwc = params


    # ARE system        
    ARE = System()

    # CORE NODES
    c_f1 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f1, name = "c_f1")
    c_f2 = Node(m = m_f_c/2, scp = scp_f, W = W_f, y0 = T0_c_f2, name = "c_f2")
    c_t1 = Node(m = m_t, scp = scp_t, y0 = T0_c_t1, name = "c_t1")
    c_c1 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c1, name = "c_c1")
    c_c2 = Node(m = m_c_c/2, scp = scp_c, W = W_c, y0 = T0_c_c2, name = "c_c2") 
    c_m1 = Node(m = m_m_c, scp = scp_m, y0 = T0_c_m + 1000, name = "c_m1")
    n = Node(y0 = n_frac0, name = "n")
    C1 = Node(y0 = C0[0], name = "C1")
    C2 = Node(y0 = C0[1], name = "C2")
    C3 = Node(y0 = C0[2], name = "C3")
    C4 = Node(y0 = C0[3], name = "C4")
    C5 = Node(y0 = C0[4], name = "C5")
    C6 = Node(y0 = C0[5], name = "C6")
    rho = Node(y0 = rho_0, name = "rho")

    # FUEL-HELIUM HX1
    hx_fh1_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1, name = "hx_fh1_f1")
    hx_fh1_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2, name = "hx_fh1_f2")
    hx_fh1_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1, name = "hx_fh1_t1")
    hx_fh1_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1, name = "hx_fh1_h1")
    hx_fh1_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2, name = "hx_fh1_h2")

    # FUEL-HELIUM HX2
    hx_fh2_f1 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f1, name = "hx_fh2_f1")
    hx_fh2_f2 = Node(m = m_f_hx/2, scp = scp_f, W = W_f/2, y0 = T0_hfh_f2, name = "hx_fh2_f2")
    hx_fh2_t1 = Node(m = m_t_hxfh, scp = scp_t, y0 = T0_hfh_t1, name = "hx_fh2_t1")
    hx_fh2_h1 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h1, name = "hx_fh2_h1")
    hx_fh2_h2 = Node(m = m_h_hxfh/2, scp = scp_h, W = W_h_fh, y0 = T0_hfh_h2, name = "hx_fh2_h2")

    # COOLANT-HELIUM HX1
    hx_ch1_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1, name = "hx_ch1_c1")
    hx_ch1_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2, name = "hx_ch1_c2")
    hx_ch1_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hch_t1, name = "hx_ch1_t1")
    hx_ch1_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1, name = "hx_ch1_h1")
    hx_ch1_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2, name = "hx_ch1_h2")

    # COOLANT-HELIUM HX2
    hx_ch2_c1 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c1, name = "hx_ch2_c1")
    hx_ch2_c2 = Node(m = m_c_hx/2, scp = scp_c, W = W_c/2, y0 = T0_hch_c2, name = "hx_ch2_c2")
    hx_ch2_t1 = Node(m = m_t_hxch, scp = scp_t, y0 = T0_hfh_t1, name = "hx_ch2_t1")
    hx_ch2_h1 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h1, name = "hx_ch2_h1")
    hx_ch2_h2 = Node(m = m_h_hxch/2, scp = scp_h, W = W_h_ch, y0 = T0_hch_h2, name = "hx_ch2_h2")

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1, name = "hx_hwf1_h1")
    hx_hwf1_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2, name = "hx_hwf1_h2")
    hx_hwf1_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1, name = "hx_hwf1_t1")
    hx_hwf1_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1, name = "hx_hwf1_w1")
    hx_hwf1_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2, name = "hx_hwf1_w2")

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h1, name = "hx_hwf2_h1")
    hx_hwf2_h2 = Node(m = m_h_hxhwf/2, scp = scp_h, W = W_h_fh, y0 = T0_hhwf_h2, name = "hx_hwf2_h2")
    hx_hwf2_t1 = Node(m = m_t_hxhwf, scp = scp_t, y0 = T0_hhwf_t1, name = "hx_hwf2_t1")
    hx_hwf2_w1 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w1, name = "hx_hwf2_w1")
    hx_hwf2_w2 = Node(m = m_w_hxhwf/2, scp = scp_w, W = W_hhwf_w, y0 = T0_hhwf_w2, name = "hx_hwf2_w2")

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1, name = "hx_hwc1_h1")
    hx_hwc1_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2, name = "hx_hwc1_h2")
    hx_hwc1_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1, name = "hx_hwc1_t1")
    hx_hwc1_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1, name = "hx_hwc1_w1")
    hx_hwc1_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2, name = "hx_hwc1_w2")

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h1, name = "hx_hwc2_h1")
    hx_hwc2_h2 = Node(m = m_h_hxhwc/2, scp = scp_h, W = W_h_ch, y0 = T0_hhwc_h2, name = "hx_hwc2_h2")
    hx_hwc2_t1 = Node(m = m_t_hxhwc, scp = scp_t, y0 = T0_hhwf_t1, name = "hx_hwc2_t1")
    hx_hwc2_w1 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w1, name = "hx_hwc2_w1")
    hx_hwc2_w2 = Node(m = m_w_hxhwc/2, scp = scp_w, W = W_hhwc_w, y0 = T0_hhwc_w2, name = "hx_hwc2_w2")



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
    
    mann = False

    # CORE
    c_f1.set_dTdt_advective(source = (hx_fh1_f2.y()+hx_fh2_f2.y())/2) 
    c_f1.set_dTdt_internal(source = [n.y()], k = [k_f1*P])
    c_f1.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])

    c_f2.set_dTdt_advective(source = c_f1.y()) 
    c_f2.set_dTdt_internal(source = [n.y()], k = [k_f2*P])
    if mann:
        c_f2.dTdt_convective = c_f1.dTdt_convective
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f1.y(), c_c1.y(), c_c1.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2]) 
    else:
        c_f2.set_dTdt_convective(source = [c_t1.y()], hA = [hA_ft_c/2])
        c_t1.set_dTdt_convective(source = [c_f1.y() ,c_f2.y(), c_c1.y(), c_c2.y()], hA = [hA_ft_c/2,hA_ft_c/2,hA_tc_c/2,hA_tc_c/2])


    c_c1.set_dTdt_advective(source = (hx_ch1_c2.y()+hx_ch2_c2.y())/2)
    c_c1.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_c2.set_dTdt_advective(source = c_c1.y())
    if mann:
        c_c2.dTdt_convective = c_c1.dTdt_convective
    else:
        c_c2.set_dTdt_convective(source = [c_t1.y(),c_m1.y()], hA = [hA_tc_c/2,hA_mc_c/2])

    c_m1.set_dTdt_internal(source = [n.y()], k = [k_m*P])
    c_m1.set_dTdt_convective(source = [c_c1.y(),c_c2.y()], hA = [hA_mc_c/2]*2)

    n.set_dndt(rho.y(), beta_t, Lam, lam, [C1.y(), C2.y(), C3.y(), C4.y(), C5.y(), C6.y()])
    # n.set_dndt(rho_0, beta_t, Lam, lam, C0)
    C1.set_dcdt(n = 1.0, beta = beta[0], Lambda = Lam, lam = lam[0], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    C2.set_dcdt(n = 1.0, beta = beta[1], Lambda = Lam, lam = lam[1], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    C3.set_dcdt(n = 1.0, beta = beta[2], Lambda = Lam, lam = lam[2], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    C4.set_dcdt(n = 1.0, beta = beta[3], Lambda = Lam, lam = lam[3], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    C5.set_dcdt(n = 1.0, beta = beta[4], Lambda = Lam, lam = lam[4], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    C6.set_dcdt(n = 1.0, beta = beta[5], Lambda = Lam, lam = lam[5], t_c = tau_c, t_l = tau_l, flow = True, force_steady_state = False)
    rho.set_drdt([c_f1.dydt,c_f2.dydt,c_m1.dydt,c_c1.dydt,c_c2.dydt],[a_f/2,a_f/2,a_b,a_c/2,a_c/2])

    # FUEL-HELIUM HX1
    hx_fh1_f1.set_dTdt_advective(source = c_f2.y())
    hx_fh1_f1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])

    hx_fh1_f2.set_dTdt_advective(source = hx_fh1_f1.y())
    if mann:
        hx_fh1_f2.dTdt_convective = hx_fh1_f1.dTdt_convective
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f1.y(),hx_fh1_h1.y(),hx_fh1_h1.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh1_f2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ft_hx/2])
        hx_fh1_t1.set_dTdt_convective(source = [hx_fh1_f1.y(),hx_fh1_f2.y(),hx_fh1_h1.y(),hx_fh1_h2.y()],
                                hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])

    hx_fh1_h1.set_dTdt_advective(source = hx_hwf2_h2.y())
    hx_fh1_h1.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    hx_fh1_h2.set_dTdt_advective(source = hx_fh1_h1.y())
    if mann:
        hx_fh1_h2.dTdt_convective = hx_fh1_h1.dTdt_convective
    else:
        hx_fh1_h2.set_dTdt_convective(source = [hx_fh1_t1.y()], hA = [hA_ht_hx/2])

    # FUEL-HELIUM HX2
    hx_fh2_f1.set_dTdt_advective(source = c_f2.y())
    hx_fh2_f1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])

    hx_fh2_f2.set_dTdt_advective(source = hx_fh2_f1.y())
    if mann:
        hx_fh2_f2.dTdt_convective = hx_fh2_f1.dTdt_convective
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f1.y(),hx_fh2_h1.y(),hx_fh2_h1.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])
    else:
        hx_fh2_f2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ft_hx/2])
        hx_fh2_t1.set_dTdt_convective(source = [hx_fh2_f1.y(),hx_fh2_f2.y(),hx_fh2_h1.y(),hx_fh2_h2.y()],
                                    hA = [hA_ft_hx/2,hA_ft_hx/2,hA_ht_hx/2,hA_ht_hx/2])


    hx_fh2_h1.set_dTdt_advective(source = hx_hwf1_h2.y())
    hx_fh2_h1.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    hx_fh2_h2.set_dTdt_advective(source = hx_fh2_h1.y())
    if mann:
        hx_fh2_h2.dTdt_convective = hx_fh2_h1.dTdt_convective
    else:
        hx_fh2_h2.set_dTdt_convective(source = [hx_fh2_t1.y()], hA = [hA_ht_hx/2])

    # COOLANT-HELIUM HX1
    hx_ch1_c1.set_dTdt_advective(source = c_c2.y())
    hx_ch1_c1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])

    hx_ch1_c2.set_dTdt_advective(source = hx_ch1_c1.y())
    if mann:
        hx_ch1_c2.dTdt_convective = hx_ch1_c1.dTdt_convective
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c1.y(),hx_ch1_h1.y(),hx_ch1_h1.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch1_c2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_ct_hx/2])
        hx_ch1_t1.set_dTdt_convective(source = [hx_ch1_c1.y(),hx_ch1_c2.y(),hx_ch1_h1.y(),hx_ch1_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch1_h1.set_dTdt_advective(source = hx_hwc1_h2.y())
    hx_ch1_h1.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    hx_ch1_h2.set_dTdt_advective(source = hx_ch1_h1.y())
    if mann:
        hx_ch1_h2.dTdt_convective = hx_ch1_h1.dTdt_convective
    else:
        hx_ch1_h2.set_dTdt_convective(source = [hx_ch1_t1.y()], hA = [hA_th_hxch/2])

    # COOLANT-HELIUM HX2
    hx_ch2_c1.set_dTdt_advective(source = c_c2.y())
    hx_ch2_c1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])

    hx_ch2_c2.set_dTdt_advective(source = hx_ch2_c1.y())
    if mann:
        hx_ch2_c2.dTdt_convective = hx_ch2_c1.dTdt_convective
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c1.y(),hx_ch2_h1.y(),hx_ch2_h1.y()],
                                hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])
    else:
        hx_ch2_c2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_ct_hx/2])
        hx_ch2_t1.set_dTdt_convective(source = [hx_ch2_c1.y(),hx_ch2_c2.y(),hx_ch2_h1.y(),hx_ch2_h2.y()],
                                    hA = [hA_ct_hx/2,hA_ct_hx/2,hA_th_hxch/2,hA_th_hxch/2])


    hx_ch2_h1.set_dTdt_advective(source = hx_hwc2_h2.y())
    hx_ch2_h1.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    hx_ch2_h2.set_dTdt_advective(source = hx_ch2_h1.y())
    if mann:
        hx_ch2_h2.dTdt_convective = hx_ch2_h1.dTdt_convective
    else:
        hx_ch2_h2.set_dTdt_convective(source = [hx_ch2_t1.y()], hA = [hA_th_hxch/2])

    # HELIUM-WATER HX1 (FUEL LOOP)
    hx_hwf1_h1.set_dTdt_advective(source = hx_fh1_h2.y())
    hx_hwf1_h1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf1_h2.set_dTdt_advective(source = hx_hwf1_h1.y())
    if mann:
        hx_hwf1_h2.dTdt_convective = hx_hwf1_h1.dTdt_convective
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h1.y(),hx_hwf1_w1.y(),hx_hwf1_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf1_h2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf1_t1.set_dTdt_convective(source = [hx_hwf1_h1.y(),hx_hwf1_h2.y(),hx_hwf1_w1.y(),hx_hwf1_w2.y()],
                                    hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])


    hx_hwf1_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf1_w1.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf1_w2.set_dTdt_advective(source = hx_hwf1_w1.y())
    if mann:
        hx_hwf1_w2.dTdt_convective = hx_hwf1_w1.dTdt_convective
    else:
        hx_hwf1_w2.set_dTdt_convective(source = [hx_hwf1_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX2 (FUEL LOOP)
    hx_hwf2_h1.set_dTdt_advective(source = hx_fh2_h2.y())
    hx_hwf2_h1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])

    hx_hwf2_h2.set_dTdt_advective(source = hx_hwf2_h1.y())
    if mann:
        hx_hwf2_h2.dTdt_convective = hx_hwf2_h1.dTdt_convective
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h1.y(),hx_hwf2_w1.y(),hx_hwf2_w1.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])
    else:
        hx_hwf2_h2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_ht_hxhw/2])
        hx_hwf2_t1.set_dTdt_convective(source = [hx_hwf2_h1.y(),hx_hwf2_h2.y(),hx_hwf2_w1.y(),hx_hwf2_w2.y()],
                                hA = [hA_ht_hxhw/2,hA_ht_hxhw/2,hA_tw_hxhw/2,hA_tw_hxhw/2])

    hx_hwf2_w1.set_dTdt_advective(source = T0_hhwf_w1)
    hx_hwf2_w1.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    hx_hwf2_w2.set_dTdt_advective(source = hx_hwf2_w1.y())
    if mann:
        hx_hwf2_w2.dTdt_convective = hx_hwf2_w1.dTdt_convective
    else:
        hx_hwf2_w2.set_dTdt_convective(source = [hx_hwf2_t1.y()], hA = [hA_tw_hxhw/2])

    # HELIUM-WATER HX1 (COOLANT LOOP)
    hx_hwc1_h1.set_dTdt_advective(source = hx_ch1_h2.y())
    hx_hwc1_h1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])

    hx_hwc1_h2.set_dTdt_advective(source = hx_hwc1_h1.y())
    if mann:
        hx_hwc1_h2.dTdt_convective = hx_hwc1_h1.dTdt_convective
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h1.y(),hx_hwc1_w1.y(),hx_hwc1_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc1_t1.set_dTdt_convective(source = [hx_hwc1_h1.y(),hx_hwc1_h2.y(),hx_hwc1_w1.y(),hx_hwc1_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc1_h2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc1_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc1_w1.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc1_w2.set_dTdt_advective(source = hx_hwc1_w1.y())
    if mann:
        hx_hwc1_w2.dTdt_convective = hx_hwc1_w1.dTdt_convective
    else:
        hx_hwc1_w2.set_dTdt_convective(source = [hx_hwc1_t1.y()], hA = [hA_tw_hxhwc/2])

    # HELIUM-WATER HX2 (COOLANT LOOP)
    hx_hwc2_h1.set_dTdt_advective(source = hx_ch2_h2.y())
    hx_hwc2_h1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2]) 

    hx_hwc2_h2.set_dTdt_advective(source = hx_hwc2_h1.y())
    if mann:
        hx_hwc2_h2.dTdt_convective = hx_hwc2_h1.dTdt_convective
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h1.y(),hx_hwc2_w1.y(),hx_hwc2_w1.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
    else:
        hx_hwc2_t1.set_dTdt_convective(source = [hx_hwc2_h1.y(),hx_hwc2_h2.y(),hx_hwc2_w1.y(),hx_hwc2_w2.y()],
                                    hA = [hA_ht_hxhwc/2,hA_ht_hxhwc/2,hA_tw_hxhwc/2,hA_tw_hxhwc/2])
        hx_hwc2_h2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_ht_hxhwc/2])


    hx_hwc2_w1.set_dTdt_advective(source = T0_hhwc_w1)
    hx_hwc2_w1.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    hx_hwc2_w2.set_dTdt_advective(source = hx_hwc2_w1.y())
    if mann:
        hx_hwc2_w2.dTdt_convective = hx_hwc2_w1.dTdt_convective
    else:
        hx_hwc2_w2.set_dTdt_convective(source = [hx_hwc2_t1.y()], hA = [hA_tw_hxhwc/2])

    # populate with analytically calculated steady-state
    for idx, key in enumerate(ARE.nodes):
        ARE.nodes[key].y0 = float(sol_list_eq[idx])

    T, sol_jit = ARE.equilibrium_search(dT = 0.01, 
                                    max_delay = tau_l, 
                                    populate_nodes = False, 
                                    md_step = 0.0001, 
                                    abs_tol_eq = 1.0e-12, 
                                    rel_tol_eq = 5.0e-8,
                                    show_conv_metrics = True)

    return sol_jit