# import numpy as np

# # SENSOR PARAMETERS (found experimentally)
# T2_PARAM = 1./(182*np.pi*2)
# R = 96. # power-spectral-density of the optical shot-noise
#
#
# LARMOR_FREQ = 2*np.pi*9999.8 # [Hz]
# PROBING_FREQ = 2*np.pi*9999.8 # [Hz]
# CORRELATION_CONSTANT_OU_PROCESS = 100 # [1/s]


# SAMPLING_PERIOD = 1./200000 #DELTA
# S_at = 118.7
# S_ph = 96.0
#
# POWER = 500e-6  # [W]
# r_e = 2.82e-15  # [m]
# f_osc = 0.34
# delta_nu = 2.4e9  # [Hz]
# c = 3e8  # [m/s]
# A_eff = 0.016e-4  # [m^2]
#
# g_D_COUPLING_CONST = 2*R*POWER*c*r_e*f_osc/(A_eff*delta_nu)
# SCALAR_STREGTH_Y = 2*S_at/(T2_PARAM*g_D_COUPLING_CONST*g_D_COUPLING_CONST)
# SCALAR_STREGTH_Z = SCALAR_STREGTH_Y

g_d_COUPLING_CONST = 50.
g_a_COUPLING_CONST = 1.
SCALAR_STREGTH_y = 0.001
SCALAR_STREGTH_j = 0.003
SCALAR_STRENGTH_q = 0.003
SCALAR_STREGTH_y_NO = 0.0
SCALAR_STREGTH_j_NO = 0.0
SCALAR_STRENGTH_q_NO = 0.0

NUM_STEPS = 100
