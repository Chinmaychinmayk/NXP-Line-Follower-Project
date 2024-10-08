import numpy as np
def PGrid(xr,yr,phi, dti, theta_ti):
    xi = xr + dti * np.cos(phi - theta_ti)
    yi = yr + dti * np.sin(phi - theta_ti)
    return xi,yi

def cost(message):
    