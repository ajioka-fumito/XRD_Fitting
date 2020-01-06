import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import numpy as np
import scipy as sp
import scipy.special

def voigt(xval,params):
    norm,center,lw,gw = params
    # norm : normalization 
    # center : center of Lorentzian line
    # lw : HWFM of Lorentzian 
    # gw : sigma of the gaussian 
    z = (xval - center + 1j*lw)/(gw * np.sqrt(2.0))
    w = scipy.special.wofz(z)
    model_y = norm * (w.real)/(gw * np.sqrt(2.0*np.pi))
    return model_y

# plot init function 
plt.title("Voigt function")
x = np.arange(0,100,0.1)

y0 = voigt(x,[1,np.mean(x),1,1])
plt.plot(x, y0/np.amax(y0), 'k-', label = "lw = 1, gw = 1")

y1 = voigt(x,[1,np.mean(x),1,10])
plt.plot(x, y1/np.amax(y1), '-', label = "lw = 1, gw = 10")

y2 = voigt(x,[1,np.mean(x),10,1])
plt.plot(x, y2/np.amax(y2), '-', label = "lw = 10, gw = 1")

y3 = voigt(x,[1,np.mean(x),10,10])
plt.plot(x, y3/np.amax(y3), '-', label = "lw = 10, gw = 10")

plt.legend(numpoints=1, frameon=False, loc="best")
plt.grid(linestyle='dotted',alpha=0.5)
plt.savefig("voigt.png")
plt.show()
