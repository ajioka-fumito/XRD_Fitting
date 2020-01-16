import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 


class SubFunctions:
    # sector 1
    def K(self):
        theta = np.array(self.params["2theta"])/2
        theta = np.pi*theta/180
        K = 2*np.sin(theta)/self.lam
        return K
    
    def delta_K(self):
        theta = np.array(self.params["2theta"])/2
        theta = np.pi*theta/180
        delta_theta = np.array(self.params["FWHM"])
        delta_theta = np.pi*delta_theta/180
        delta_K = 2*np.cos(theta)*delta_theta/self.lam # 掛ける２すべき？
        return delta_K
    
    #sector 2
    def H2(self):
        H2 = []
        for orientation in self.orientations:
            h,k,l = orientation
            h,k,l = int(h),int(k),int(l)
            h2 = ((h*k)**2+(h*l)**2+(k*l)**2)/(h**2+k**2+l**2)**2
            H2.append(h2)
        return np.array(H2)
    
    # sector 3
    def alpha(self):
        x = self.H2
        coef = []
        for i in range(0,10**4):
            alpha = 2*i*10**(-7)
            y = (self.delta_K**2-alpha)/self.K**2
            now = np.corrcoef(x,y)[1][0]
            coef.append(now**2)
        idx = np.argmax(coef)
        alpha = 2*idx*10**(-7)
        return alpha

    def A_L(self):
        A_L = pd.DataFrame(columns=self.orientations)
        for orientation in self.orientations:
            theta = np.array(self.fitting[orientation+"thetas"])
            delta_theta = (2*(np.sin(theta[-1]*np.pi/360)-np.sin(theta[0]*np.pi/360))/self.lam)
            intensity = self.fitting[orientation+"intensity"]/np.max(self.fitting[orientation+"intensity"])
            # ここが必要か不明
            """
            idx = np.argmax(intensity)
            wid = np.min([1000-idx,idx])
            intensity = intensity[idx-wid:idx+wid]
            #plt.plot(np.arange(len(intensity)),intensity)
            """
            fft = abs(np.fft.fft(intensity))
            fft = fft/np.max(fft)
            delta_hlz = 1/delta_theta
            hlz = np.arange(0,delta_hlz*len(fft)-0.001,delta_hlz)
            tck = interpolate.splrep(hlz,fft)
            fft_sp = interpolate.splev(np.arange(51),tck)
            A_L[orientation] = fft_sp
            plt.plot(np.arange(51),fft_sp)

        if os.path.exists(self.out_dir)!=1:
            os.makedirs(self.out_dir)
        plt.savefig(self.out_dir+"/ALvsL.png")
        plt.clf()
        return A_L

    def fffit(self):
        ls_b,ls_c = [],[]
        for i in range(1,51):
            _,b,c = np.polyfit(self.K2C,self.lnA_L.loc[i],2)
            plt.scatter(self.K2C,self.lnA_L.loc[i])
            if i%2==0:
                ls_b.append(b)
                ls_c.append(c)

        plt.savefig(self.out_dir+"/ALvsK2C.png")
        plt.clf()
        return np.array(ls_c),np.array(ls_b) 
    
    def Re(self):
        # ↓　ここでフィッテイングに用いる点を制御
        slp,inter = np.polyfit(self.ln_L[4:8],self.X_L_L2[4:8],1)

        plt.scatter(self.ln_L,self.X_L_L2)
        plt.scatter(self.ln_L[4:8],self.X_L_L2[4:8])
        plt.plot(np.arange(0,4,0.01),slp*np.arange(0,4,0.01)+inter)
        plt.ylim(-0.0014,0)
        plt.savefig(self.out_dir+"/X_L_L2vslnL.png")
        plt.clf()
        return slp,inter


    def D(self):
        a,b = np.polyfit(np.arange(2,10,2),self.As_L[1:5],1)

        plt.scatter(np.arange(2,10,2),self.As_L[1:5])
        plt.scatter(np.arange(2,51,2),self.As_L)
        plt.plot(np.arange(2,10,2),a*np.arange(2,10,2)+b)
        plt.savefig(self.out_dir+"/AsLvsL.png")
        plt.clf()
        return -b/a


class Main(SubFunctions):
    def __init__(self,predict_dir,output_dir,orientations,remove_orientations,ka_lambda,C_h00):
        super().__init__()
        self.dir = predict_dir
        self.out_dir = output_dir
        self.orientations = orientations 
        self.remove_orientation(remove_orientations)
        self.crete_data()
        self.params = pd.read_csv(self.dir+"/dataset/2theta_FWHM.csv")
        self.fitting = pd.read_csv(self.dir+"/dataset/fitting.csv")

        # sector 1 calculate K and delta_k
        self.lam = ka_lambda
        self.K,self.delta_K  = self.K(),self.delta_K()

        # sector 2 calculate H_2
        self.C_h00 = C_h00
        self.H2    = self.H2()

        # sector 3 calculate q and C_bar
        self.alpha = self.alpha()
        self.beta_B,self.beta_A = np.polyfit(self.H2,(self.delta_K**2-self.alpha)/self.K**2,1)
        self.q = -self.beta_B/self.beta_A
        self.C_bar = self.C_h00*(1-self.q*self.H2)
        
        # sector 4 calculate A_L using fourie
        self.A_L = self.A_L()
        self.lnA_L = self.A_L.apply(np.log)
        

        # sector 5 calculate AsL and X_L
        self.K2C = self.K**2*self.C_bar
        self.lnAs_L,self.X_L = self.fffit() # correspond L (0~50)  
        self.As_L = np.exp(self.lnAs_L)
        self.ln_L = np.log(np.arange(2,51,2)) # correspond L (2~50,2)
        self.X_L_L2 = self.X_L/np.arange(2,51,2)**2
        
        # sector 6 calculate rho
        self.slp,self.inter = self.Re()
        self.Re = np.exp(-self.inter/self.slp)
        self.Re_ = self.Re/7.39
        self.rho = 2*self.slp*10**20/(np.pi*2.5**2)
        self.M = self.rho**(0.5)*self.Re_*10**(-9)
        

        # sector 6 calculate D
        self.D = self.D() # ok
        print("D:",self.D)
        print("rho:","{:.3e}".format(self.rho))
        print("M:",self.M)
        print("q:",self.q)
    
    def remove_orientation(self,remove_orientations):
        for now in remove_orientations:
            self.orientations.remove(now)

    def crete_data(self):
        df_2theta_FWHM = pd.DataFrame({"2theta":[],"FWHM":[]})
        for orientation in self.orientations:
            data = pd.read_csv(self.dir+"/"+orientation+".csv")
            data = data.loc[0,["2theta","FWHM"]]
            df_2theta_FWHM.loc[orientation] = data
        df_thetas_intensity = pd.DataFrame()
        for orientation in self.orientations:
            data = pd.read_csv(self.dir+"/"+orientation+".csv")
            data = data.loc[:,["thetas","intensity"]]
            df_thetas_intensity[orientation+"thetas"] = data["thetas"]
            df_thetas_intensity[orientation+"intensity"] = data["intensity"]
        if os.path.exists(self.dir+"/dataset") == False:
            os.makedirs(self.dir+"/dataset")

        df_2theta_FWHM.to_csv(self.dir+"/dataset/2theta_FWHM.csv")
        df_thetas_intensity.to_csv(self.dir+"/dataset/fitting.csv",index=False)
        print("Created Data Set")

if __name__ == "__main__":

    predict_dir = "./data/fitting/output/predict/lorentz"
    output_dir = "./data/modified_WA/output/graphs"
    # ["110","200","211","220","310","222"]
    orientations = ["110","200","211","220","310","222"]
    remove_orientations = ["110"]
    
    ins = Main(predict_dir = predict_dir,
               output_dir = output_dir,
               orientations = orientations,
               remove_orientations = remove_orientations,
               ka_lambda = 0.070931,
               C_h00 = 0.285)

