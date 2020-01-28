import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 
from scipy.special import wofz

class SubFunctions:    
    def max_intensity(self):
        """
        Caluculate the peak of ka1
        the angle corresponding to the maxmun intensity
        """
        idx = np.argmax(np.array(self.t))
        return self.x[idx],self.t[idx]

    def smooth(self):
        fft = np.fft.fft(self.t)
        nor = np.max(fft)
        fft_nor = fft/nor
        fft_nor = np.where(abs(fft_nor)<0.01,0,fft_nor)
        t_nor = abs(np.fft.ifft(nor*fft_nor))
        return t_nor

    def bragg(self,*params):
        """
        params = [lamda_ka1,lamda_ka2,theta1,delta]
        """
        theta1 = params[2]*np.pi/180/2
        delta  = params[3]*np.pi/180
        return abs(params[1]/params[0]*np.sin(theta1)-np.sin(theta1+delta))

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
        noise = sorted(self.t)
        noise = noise[:150]
        return min(noise) 

    def create_init_params(self):
        
        if self.function == "gauss":
            sigma = 0.08
            params = [self.noise,
                      0.8*self.t_max,self.x1,sigma,
                      0.8*0.4*self.t_max,self.x1+self.delta_x,sigma*1.1]
            return params

        elif self.function == "lorentz":
            ganma = 0.04
            params = [self.noise,
                      0.8*self.t_max,self.x1,ganma,
                      0.8*0.4*self.t_max,self.x1+self.delta_x,ganma*1.3]
            return params

        elif function == "voigt":
            ganma,sigma = 0.05,0.01
            params = [self.noise,
                      0.8*self.t_max,self.x1,ganma,sigma,
                      0.8*0.4*self.t_max,self.x1+self.delta_x,ganma*1.3,sigma*1.3]
            return params
        else:
            print("input function does not exist.")
            exit()


class FittingFunctions:
    """
    functions for fitting
    """

    def gauss(self,x,params):
        """
        params = [h,x0,sigma]
        """
        y = np.exp(-(x-params[1])**2/params[2]**2)
        y = y/np.max(y)
        y = params[0]*y
        return y

    def lorentz(self,x,params):
        """
        params = [h,x0,ganma]
        """
        y = 1/(np.pi*params[2]) * (params[2]**2/((x-params[1])**2 + params[2]**2))
        y = y/np.max(y)
        y = params[0]*y
        return y

    def voigt(self,x,params):
        """
        params = [h,x0,ganma,sigma]
        """
        z = ((x-params[1])+1j*params[2]) /(params[3]*np.sqrt(2.0*np.pi))
        w = wofz(z)
        v = (w.real / (params[3] * np.sqrt(2.0)))
        v = v/np.max(v)
        v = params[0]*v
        return v

    def stack_funcs(self,x,*params):
        noize = params[0]
        ka1_params = params[1:(len(params)+1)//2]
        ka2_params = params[(len(params)+1)//2:]

        if self.function == "gauss":
            stack = self.gauss(x,ka1_params)+self.gauss(x,ka2_params)+noize
            return stack

        elif self.function == "lorentz":
            stack = self.lorentz(x,ka1_params)+self.lorentz(x,ka2_params)+noize
            return stack

        elif self.function == "voigt":
            stack = self.gauss(x,ka1_params)+self.gauss(x,ka2_params)+noize
            return stack
        else:
            print("input function does not exist.")
            exit()



class Main(SubFunctions,FittingFunctions):
    def __init__(self,path,file_name,orientation,kalpha,function,output_dir):
        # constant
        self.ka1, self.ka2 = kalpha # dim:angstrom
        self.function = function
        # data
        self.file_name = file_name
        self.output_dir = output_dir
        self.data = pd.read_csv(path)
        self.orientation = orientation
        self.x,self.t = self.data[self.orientation+"theta"],self.data[self.orientation+"intensity"]
        self.x = self.x.dropna(how="all")
        self.t = self.t.dropna(how="all")
        self.t_nor = self.smooth()

        if (len(self.x)==0 or len(self.t)==0):
            print("read file is enmpy or wrong path")
            exit()

        # caluculated params
        self.x1,self.t_max = self.max_intensity()
        """
        self.smooth()
        """
        self.delta_x = 2*self.delta() # delta() return delta theta in bragg 
        self.noise = self.Noise()

        # init params for curve fitting
        self.init_params = self.create_init_params()
        # fitted gauss params 
        self.popt = self.fitting()
        print("fiited_params:",self.popt)
    def fitting(self):
        popt,_ = curve_fit(self.stack_funcs,self.x,self.t_nor,p0= self.init_params)
        return popt

    def output(self):
        
        if os.path.exists(self.output_dir) == False:
            os.makedirs(self.output_dir)
            os.makedirs(self.output_dir+"/figs")
        if self.function == "gauss":
            predict1 = self.gauss(self.x,self.popt[1:4])
            predict2 = self.gauss(self.x,self.popt[4:])
            predict_all = self.stack_funcs(self.x,*self.popt)
            df1 = pd.DataFrame({"noise":[self.popt[0]],"a_ka1":[self.popt[1]],"b_ka2":[self.popt[2]],"c_ka2":[self.popt[3]],
                                "2theta":[self.popt[2]],"FWHM":[2*np.log(2)*self.popt[3]]})
        elif self.function == "lorentz":
            predict1 = self.lorentz(self.x,self.popt[1:4])
            predict2 = self.lorentz(self.x,self.popt[4:])
            predict_all = self.stack_funcs(self.x,*self.popt)
            df1 = pd.DataFrame({"noise":[self.popt[0]],"h":[self.popt[1]],"x0":[self.popt[2]],"ganma":[self.popt[3]],
                                "2theta":[self.popt[2]],"FWHM":2*self.popt[3]})

        elif self.function == "voigt":
            predict1 = self.voigt(self.x,self.popt[1:5])
            predict2 = self.voigt(self.x,self.popt[5:])
            predict_all = self.stack_funcs(self.x,*self.popt)
            f_g = 2*self.popt[4]*np.sqrt(2*np.log(2))
            f_l = 2*self.popt[3]
            df1 = pd.DataFrame({"noise":[self.popt[0]],"h":[self.popt[1]],"x0":[self.popt[2]],"ganma":[self.popt[3]],"sigma":[self.popt[4]],
                                "2theta":[self.popt[2]],"FWHM":[0.5346*f_l+np.sqrt(0.2166*f_l**2+f_g)]})
        else:
            print("input function does not exist.")
            exit()
        df2 = pd.DataFrame({"thetas":self.x,"intensity":predict1})
        df  = pd.concat([df1,df2],axis=1)
        df.to_csv(self.output_dir+"/"+self.orientation+".csv",index=None)
        
        plt.plot(self.x,self.t,color="darkgreen")
        plt.plot(self.x,self.t_nor,color="yellow")
        plt.plot(self.x,predict_all,color="red")
        plt.fill_between(self.x,predict1,color="purple")
        plt.fill_between(self.x,predict2,color="blue")
        plt.savefig(self.output_dir+"/figs/"+self.orientation+".png")
        plt.clf()

if __name__ == "__main__":
    # initial params (you dont have to input)
    orientations = ["111","200","220","311","222","400"]

    # initial params (you have to input)
    # file_name 
    file_name = "peaks.csv"
    # ka lamda
    ka = [0.70926, 0.71354]
    # fitting functon {"gauss","lorentz","voigt"}
    function = "lorentz"
    output_dir = "./data/fitting/output/predict/"+function
    for orientation in orientations:
        # check fitting 
        ins = Main(path="./data/fitting/input/"+file_name,
                   file_name=file_name,
                   orientation = orientation, kalpha=ka, function=function,
                   output_dir = output_dir)
        # to csv
        ins.output()