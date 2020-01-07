import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit 

class SubFunctions:
    def bragg(self,*params):
        """
        params = [lamda_ka1,lamda_ka2,theta1,delta]
        """
        theta1 = params[2]*np.pi/180/2
        delta  = params[3]*np.pi/180
        return abs(params[1]/params[0]*np.sin(theta1)-np.sin(theta1+delta))

    def max_intensity(self):
        """
        Caluculate the peak of ka1
        the angle corresponding to the maxmun intensity
        """
        idx = np.argmax(self.t)
        return self.x[idx],self.t[idx]
    
    def delta(self):
        """
        delta theta that can be caluculated by theta_ka1
        """
        # init paramas
        ans = 10**9
        delta = 0
        for i in range(0,1000,2):
            now = i/1000
            sub = self.bragg(*[self.ka1,self.ka2,self.x1,now])
            if sub<ans:
                delta = now
                ans = sub
        return delta

    def Noise(self):
        noise1 = np.mean(self.t[0:50])
        noise2 = np.mean(self.t[len(self.t)-51:len(self.t)-1])
        return (noise1+noise2)/2

class FittingFunctions:
    """
    functions for fitting
    """

    def lorentz(self,x,*params):
        """
        params = [h,x0,ganma]
        """
        y = 1/(np.pi*params[2]) * (params[2]**2/((x-params[1])**2 + params[2]**2))
        y = y/np.max(y)
        y = params[0]*y
        return y


class Visualize:
    def __init__(self):
        pass

    def plot(self):
        predict = self.lorentz_plus(self.x,*self.popt)
        plt.plot(self.x,self.t)
        plt.plot(self.x,predict)
        for i in range(2):
            predict = self.lorentz(self.x,*self.popt[1+3*i:4+3*i])
            plt.fill_between(self.x,predict,facecolor=cm.rainbow(i/2, alpha=0.6))
            
        plt.savefig("./data/fitting/output/figs/"+self.orientation+".png")
        plt.clf()


class Main(FittingFunctions,SubFunctions,Visualize):
    def __init__(self,path,orientation):
        super().__init__()
        # constant
        self.ka1, self.ka2 = 0.70926, 0.71354 # dim:angstrom
        # data
        self.data = pd.read_csv(path)
        self.orientation = orientation
        self.x,self.t = self.data[orientation+"theta"],self.data[orientation+"intensity"]
        # caluculated params
        self.x1,self.t_max = self.max_intensity()
        self.delta_x = self.delta()*2
        self.noise = self.Noise()
        # init params for curve fitting
        self.init_params = [self.noise,
                            0.8*self.t_max,self.x1,0.04,
                            0.8*0.4*self.t_max,self.x1+self.delta_x,0.04*1.3]
        # fitted gauss params 
        self.popt = self.fitting()
        

    def lorentz_plus(self,x,*params):
        y_ka1 = self.lorentz(x,*params[1:4])
        y_ka2 = self.lorentz(x,*params[4:7])
        y = y_ka1+y_ka2+self.init_params[0]
        return y

    def fitting(self):
        popt,_ = curve_fit(self.lorentz_plus,self.x,self.t,p0=self.init_params)
        return popt


        


    def out(self,filename):
        predict = self.lorentz(self.x,*self.popt[1:4])

        df1 = pd.DataFrame({"noise":[self.popt[0]],"h":[self.popt[1]],"x0":[self.popt[2]],"ganma":[self.popt[3]],
                            "2theta":[self.popt[2]],"FWHM":2*self.popt[3]})
        
        df2 = pd.DataFrame({"thetas":self.x,"intensity":predict})
        df  = pd.concat([df1,df2],axis=1)
        df.to_csv(filename,index=None)
 
if __name__ == "__main__":
    # initial params
    orientations = ["110","200","211","220","310","222"]

    for orientation in orientations:
        # check fitting 
        ins = Main("./data/fitting/input/peaks.csv",orientation)
        ins.plot()
        # to csv
        ins.out("./data/fitting/output/predicts/"+orientation+".csv")
