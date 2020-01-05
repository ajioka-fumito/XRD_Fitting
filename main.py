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
    def __init__(self):
        pass

    def gauss(self,x,*params):
        """
        Ka_params : [max_intensity(a),position(b),width(c)] vector
        back_ground : scalar
        y = a * exp(-(x-b)**2/c**2)
        """
        y = params[0]*np.exp(-(x-params[1])**2/params[2]**2)
        return y

    def gauss_plus(self,x,*params):
        """
        Ka_params : [max_intensity(a),position(b),width(c)] vector
        back_ground : scalar
        y = a * exp(-(x-b)**2/c**2)
        """

        y_ka1 = params[1]*np.exp(-(x-params[2])**2/params[3]**2)
        y_ka2 = params[4]*np.exp(-(x-params[5])**2/params[6]**2)
        y = y_ka1+y_ka2+params[0]
        return y

class Visualize:
    def __init__(self):
        pass

    def plot(self):
        predict = self.gauss_plus(self.x,*self.popt)
        plt.plot(self.x,self.t)
        plt.plot(self.x,predict)
        for i in range(2):
            predict = self.gauss(self.x,*self.popt[1+3*i:4+3*i])
            plt.fill_between(self.x,predict,facecolor=cm.rainbow(i/2, alpha=0.6))
        plt.savefig("./data/output/figs/"+self.orientation+".png")
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
                            0.8*self.t_max,self.x1,0.08,
                            0.8*0.4*self.t_max,self.x1+self.delta_x,0.08*1.1]
        # fitted gauss params 
        self.popt = self.fitting()
        
    def fitting(self):
        popt,_ = curve_fit(self.gauss_plus,self.x,self.t,p0=self.init_params)
        return popt

    def out(self,filename):
        predict = self.gauss(self.x,*self.popt[1:4])
        
        df1 = pd.DataFrame({"noise":[self.popt[0]],"a_ka1":[self.popt[1]],"b_ka2":[self.popt[2]],"c_ka2":[self.popt[3]],
                            "2theta":[self.popt[2]],"FWHM":[2*np.log(2)*self.popt[3]]})

        df2 = pd.DataFrame({"thetas":self.x,"intensity":predict})
        df  = pd.concat([df1,df2],axis=1)
        df.to_csv(filename,index=None)

if __name__ == "__main__":
    # initial params
    orientations = ["110","200","211","220","310","222"]

    for orientation in orientations:
        # check fitting 
        ins = Main("./data/input/peaks.csv",orientation)
        ins.plot()
        # to csv
        ins.out("./data/output/"+orientation+".csv")

