#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:13:48 2020

@author: JeremiahDavid
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

plt.close('all')

#%% Import the data
data = pd.read_excel('Engine_Specs.xlsx',skiprows=[0,1,2])
data.columns = ['EngineName', 'Displacement_L', 'Cylinders', 'PeakPower_HP',
                'RPM', 'CompRatio', 'Stroke_mm', 'Source']
data.sort_values(by='CompRatio',inplace=True)

def cp_eq(T):
    # equation to estimate the soecific heat (pressure) 
        # equation obtained from ME4011 course work
    theta = T/100
    cp0_N2 = 39.060 - 512.79*(theta)**-1.5 + 1072.7*(theta)**-2 - 820.40*(theta)**-3
    cp0_O2 = 37.434 + 0.0201*(theta)**1.5 - 178.57*(theta)**-1.5 + 236.88*(theta)**-2.0
    
    cpq = (0.79/28)*cp0_N2 + (0.21/32)*cp0_O2
    
    return cpq

def mech_eff(Up):
    Up = np.array(Up).reshape(-1,1) # reshape the data
    # the data taken from Pulkrabek
    piston_speed = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)
    mechanical_efficiency = np.array([92.1,88.6,85,81.25,77.5,72.9,68,62.7,57.2,50.2]).reshape(-1,1)
    # smooth set of x values to plug into the model for the plot
    piston_speed_range = np.arange(2,23,18/1000).reshape(-1,1)
    # create the regression model
    eta_m_model = LinearRegression()
    poly_reg = PolynomialFeatures(degree = 3)
    piston_speed_poly = poly_reg.fit_transform(piston_speed)
    eta_m_model.fit(piston_speed_poly,mechanical_efficiency)
    mechanical_efficiency_interpolation = eta_m_model.predict(poly_reg.fit_transform(piston_speed_range))
    mechanical_efficiency_predict = eta_m_model.predict(poly_reg.fit_transform(Up))
    # plot the data that trained the model, the regression line, and the data to be predicted
    plt.figure(1)
    plt.plot(piston_speed_range,mechanical_efficiency_interpolation,color='r',
             dashes=[2,1,2,1],linewidth=1,label='Interpolated Data')
    plt.scatter(piston_speed,mechanical_efficiency,color='b',marker='o',
                label='Raw Data')
    plt.scatter(Up,mechanical_efficiency_predict,color='k',marker='*',
                label='Predicted Mechanical Efficiency')
    plt.legend()
    plt.xlabel('Average Piston Speed [m/s]')
    plt.ylabel('Mechanical Efficiency %')
    
    return mechanical_efficiency_predict
    
# =============================================================================
# Main Program
# =============================================================================
def Engine_analysis(data,eta_c,pv_plots):
    Vd_total = data.Displacement_L # total displacement for the engine [L]
    nc = data.Cylinders # number of cylinders
    rc = data.CompRatio # compression ratio
    N = data.RPM # RPM
    stroke = data.Stroke_mm/1000
    stroke.columns = ['Stroke_m']
    bore = np.sqrt((Vd_total/1000)/(nc*stroke*(np.pi/4))) # equ. 9 in Ch.2 Pulkrabek
    
    n = 2 # rev/cycle (4 cylinder)
    
    Vs = (Vd_total/nc)/1000 # swept volume for a cylinder [m^3]
    Vc = (Vs/(rc-1)) # clearance volume for a cylinder
    
    # eta_c = 0.96 # combustion efficiency (usually this is between .95 and .98)
    xr = 0.06 # exhaust residual percentage
    eta_s = 1-xr # the ratio being multiplied later on
    
    # fuel analysis values
    LHV = 44300 #[kJ/kg]
    psy = 1.1
    # psy = 0.9 # equivilance ratio
    AF_stoich = 15
    AF = AF_stoich/psy
    
    # intake state
    p1 = 101.325 # intake pressure [kPa]
    T1 = 298.15 # [K]
    V1 = Vc + Vs
    
    k = 1.35  # Ratio of specific heat
    R = 0.287 # Gas constant for air [kJ/kg]
    
    m1 = p1*V1/(R*T1) # Ideal gas law
    
    m_m = m1 # mass of the mixture
    
    m_a = eta_s*m_m*(AF/(1+AF)) # mass of the air
    m_f = eta_s*m_m*(1/(1+AF))
    
    # Start analysis 
    # Process 1-2: Compression Stroke (Isentropic Compression)
    p2 = p1*rc**k
    T2 = T1*rc**(k-1)
    V2 = Vc
    
    W12 = m_m*R*(T2-T1)/(1-k) # work during compression
    
    # specific heat constants
    cp = cp_eq(T1) # pressure
    cv = cp - R # volume
    
    # Process 2-3: Combustion (Contant Volume)
    Qc = m_f*LHV*eta_c # heat during combustion
    T3 = T2 + Qc/(m_m*cv)
    p3 = p2 * (T3/T2)
    V3 = V2
    
    # Process 3-4: Power Stroke (Isentropic Expansion)
    p4 = p3*(rc)**-k
    T4 = T3*(rc)**-(k-1)
    V4 = V1
    
    W34 = m_m*R*(T4-T3)/(1-k) # Work during expansion
    
    # Net work
    W_net = (W12 + W34)
    
    # Blowdown
    Qout = m_m*cv*(T1-T4)
    
    # Exhaust
    p5 = p1
    V5 = Vs
    W56 = p5*(Vs-Vc)
    
    # Intake
    p6 = p1
    V6 = Vc
    W61 = p5*0.8*(Vc-Vs)
    
    # imep
    imep = W_net/(Vd_total*10**-3)
    
    # total engine power
    Power_indicated = nc*(W_net*N)/(60*n)
    Power_brake = data.PeakPower_HP*0.7457
    
    A_p = bore**2*(np.pi/4) # piston head area
    u_p = 2*stroke*(N/60) # mean piston speed
    # Power2 = (2.04e3*A_p*u_p)/4 # power
    # W_net2 = 2.04e3*(Vs-Vc)
    mech_eff_est = pd.Series(list(mech_eff(u_p)))
    mech_eff_calc = 100*Power_brake/Power_indicated
    
    # plot mech eff vs. average piston speed to check for relationship
    plt.figure(2)
    plt.scatter(u_p,mech_eff_calc,color='b',marker='*',
              label='Actual Mechanical Efficiency')
    plt.xlim((15,23))
    plt.xlabel('Average Piston Speed')
    plt.ylabel('Mechanical Efficiency %')
    
    # plot mech eff vs. comp ratio 
    plt.figure(3)
    plt.scatter(rc,mech_eff_calc,color='b',marker='*',label='Raw Data')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Mechanical Efficiency %')
    # since there is a decent relationship, develop model 
    rc_plot = np.arange(9.5,13.5,0.01).reshape(-1,1)
    rc_model = LinearRegression()
    poly_reg_rc = PolynomialFeatures(degree = 1)
    rc_poly = poly_reg_rc.fit_transform(np.array(rc).reshape(-1,1))
    rc_model.fit(rc_poly,mech_eff_calc)
    mech_eff_regression = rc_model.predict(poly_reg_rc.fit_transform(rc_plot))
    mech_eff_pred = rc_model.predict(rc_poly)
    # plot the bounds with a 94.5% confidence interval
    K = 2 # coverage factor (2 for 95.4% CI)
    error = np.sqrt(mean_squared_error(mech_eff_calc,mech_eff_pred))
    lower_bound = rc_model.coef_[1]*rc_plot + rc_model.intercept_ - K*error
    upper_bound = rc_model.coef_[1]*rc_plot + rc_model.intercept_ + K*error
    plt.plot(rc_plot,mech_eff_regression,color='r',
              dashes=[2,1,2,1],linewidth=1,label='Interpolated Data')
    plt.plot(rc_plot,lower_bound,color= 'g',
              dashes=[1,1,1,1],linewidth=1,label='95.45% Confidence Bound')
    plt.plot(rc_plot,upper_bound,color= 'g',
              dashes=[1,1,1,1],linewidth=1)
    plt.legend()
    
  
    # plot the mech eff vs. various comp ratios to see if there are correlations
    mask8 = (rc >= 8) & (rc < 9)
    Power_brake8 = Power_brake[mask8]
    mech_eff8 = mech_eff_calc[mask8]
    
    mask9 = (rc >= 9) & (rc < 10)
    Power_brake9 = Power_brake[mask9]
    mech_eff9 = mech_eff_calc[mask9]
    
    mask10 = (rc >= 10) & (rc < 11);
    Power_brake10 = Power_brake[mask10]
    mech_eff10 = mech_eff_calc[mask10]
    
    mask11 = (rc >= 11) & (rc < 12)
    Power_brake11 = Power_brake[mask11]
    mech_eff11 = mech_eff_calc[mask11]
    
    mask12 = (rc >= 12) & (rc < 13)
    Power_brake12 = Power_brake[mask12]
    mech_eff12 = mech_eff_calc[mask12]
    
    plt.figure(4)
    plt.scatter(Power_brake8,mech_eff8,label='Comp Ratio: 8')
    plt.scatter(Power_brake9,mech_eff9,label='Comp Ratio: 9')
    plt.scatter(Power_brake10,mech_eff10,label='Comp Ratio: 10')
    plt.scatter(Power_brake11,mech_eff11,label='Comp Ratio: 11')
    plt.scatter(Power_brake12,mech_eff12,label='Comp Ratio: 12')
    plt.legend()
    plt.xlabel('Brake Power')
    plt.ylabel('Mechanical Efficiency %')
    
    print('rmse: %s' %(error))
    print('gain: %s' %(rc_model.coef_[1]))
    print('intercept: %s' %(rc_model.intercept_))

    if pv_plots:
        # plots for pressure vs. volume for strokes 1-4 (all but intake and exhaust)
        for i in np.arange(0,len(p2)):
            
            print("p1: %.2f [kpa]\t\tT1: %.2f [K]" %(p1,T1))
            print("p2: %.2f [kpa]\t\tT2: %.2f [K]" %(p2[i],T2[i]))
            print("p3: %.2f [kpa]\t\tT3: %.2f [K]" %(p3[i],T3[i]))
            print("p4: %.2f [kpa]\t\tT4: %.2f [K]" %(p4[i],T4[i]))
            
            print("Net work output (indicated):\t %.2f [kJ per cycle]" %(W_net[i]))
            # print("Net work output:\t %.2f [kJ per cycle]" %(W_net2[i]))
            print("Total engine power output (indicated):\t%.2f [kW]" %(Power_indicated[i]))
            print("Total engine power output (brake):\t%.2f [kW]" %(Power_brake[i]))
            # print("Total engine power output (indicated):\t%.2f [kW]" %(Power2[i]))
        
            p = lambda v,T: m_m[i]*R*T/v # function to get pressure
            V12 = np.linspace(V1[i],V2[i],101) # range: [1-rc]
            T12 = np.linspace(T1,T2[i],101)
            p12 = p(V12,T12)
            # p12 = p1*np.divide(V12,V2)**k # a vector of (p2 = p1*rc^k) 
            
            V23 = [V2[i], V2[i]] # constant volume 
            p23 = [p2[i], p3[i]] 
            
            # p34 = p3*np.divide(V12,V2)**-k # a vector of (p2 = p1*rc^-k) 
            V34 = np.linspace(V3[i],V4[i],101) # range: [1-rc]
            T34 = np.linspace(T3[i],T4[i],101)
            p34 = p(V34,T34)
            
            V45 = [V1[i], V1[i]] # constant volume 
            p45 = [p4[i], p5]
            
            plt.figure()
            plt.plot(np.linspace(V1[i],V2[i],101),p12,'-b')
            plt.plot(V23,p23,1000,'-b')
            plt.plot(np.linspace(V2[i],V1[i],101),p34,'-b')
            plt.plot(V45,p45,'-b')
            plt.xlabel('Volume [m^3]')
            plt.ylabel('Pressure [kPa]')
        
    return mech_eff_calc

# run analysis with combustion effecieny of 90%
eta_c = 0.9
pv_plots = False
mech_effs_90 = Engine_analysis(data,eta_c,pv_plots)








