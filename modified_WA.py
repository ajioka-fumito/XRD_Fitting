
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 

class Main:
    def __init__(self):
        self.dir = "./data/fitting/output/predicts"
        # ["110","200","211","220","310","222"]
        self.orientations = ["200","211","220","310","222"] # remove 110 211 222
        self.crete_data()
        self.params = pd.read_csv("./data/modified_WA/2theta_FWHM.csv")
        self.fitting = pd.read_csv("./data/modified_WA/fitting.csv")
        # sector 1 ok
        self.lam = 0.070931
        self.K   = self.K()
        self.delta_K = self.delta_K()
        
        # sector 2 ok
        self.ch00 = 0.285
        self.H2    = self.H2()

        # sector 3
        self.alpha = self.alpha() # ok
        
        self.beta_B,self.beta_A = np.polyfit(self.H2,(self.delta_K**2-self.alpha)/self.K**2,1) # this point is wrong

        self.q = -self.beta_B/self.beta_A

        self.C_bar = self.ch00*(1-self.q*self.H2) # ok
        
        
        
        
        # sector 4 fourie
        self.A_L = self.A_L()
        self.lnA_L = self.A_L.apply(np.log)
        
        # こっから怪しい
        # sector 5
        self.K2C = self.K**2*self.C_bar
        self.lnAs_L,self.X_L = self.fffit()
        self.As_L = np.exp(self.lnAs_L)
        self.ln_L = np.log(np.arange(2,51,2))

        self.X_L_L2 = self.X_L/np.arange(2,51,2)**2
        
        self.slp,self.inter = self.Re()
        self.Re = np.exp(-self.inter/self.slp)
        self.Re_ = self.Re/7.39
        self.rho = 2*self.slp*10**20/(np.pi*2.5**2)
        self.M = self.rho**(0.5)*self.Re_*10**(-9)
        
        plt.scatter(self.ln_L,self.X_L_L2)
        plt.scatter(self.ln_L[12:16],self.X_L_L2[12:16])
        plt.plot(self.ln_L[12:16],self.slp*self.ln_L[12:16]+self.inter)
        plt.ylim(-0.0025,0)
        plt.show()

        self.D = self.D() # ok
        print("D:",self.D)
        print("rho:",self.rho)
        print("M:",self.M)
        print("q:",self.q)
        
    def Re(self):
        slp,inter = np.polyfit(self.ln_L[12:16],self.X_L_L2[12:16],1)
        return slp,inter

    
    def A_L(self):
        A_L = pd.DataFrame(columns=self.orientations)
        for orientation in self.orientations:
            theta = np.array(self.fitting[orientation+"thetas"])
            #print(theta)
            delta_theta = (2*(np.sin(theta[-1]*np.pi/360)-np.sin(theta[0]*np.pi/360))/self.lam)
            #print("delta_theta",delta_theta)
            intensity = self.fitting[orientation+"intensity"]/np.max(self.fitting[orientation+"intensity"])
            
            # ここが必要か不明
            """
            idx = np.argmax(intensity)
            intensity = intensity[idx-250:idx+251]
            plt.plot(np.arange(len(intensity)),intensity)
            """
            fft = abs(np.fft.fft(intensity))
            fft = fft/np.max(fft)
            delta_hlz = 1/delta_theta
            hlz = np.arange(0,delta_hlz*len(fft)-0.001,delta_hlz)
            tck = interpolate.splrep(hlz,fft)
            fft_sp = interpolate.splev(np.arange(51),tck)
            A_L[orientation] = fft_sp
            """
            plt.plot(np.arange(51),fft_sp)
            plt.show()
            """
        return A_L

    def D(self):
        a,b = np.polyfit(np.arange(2,10,2),self.As_L[0:4],1)
        plt.scatter(np.arange(2,10,2),self.As_L[0:4])
        plt.plot(np.arange(2,10,2),a*np.arange(2,10,2)+b)
        plt.show()
        return -b/a


    def fffit(self):
        ls_b,ls_c = [],[]
        for i in range(1,51):
            _,b,c = np.polyfit(self.K2C,self.lnA_L.loc[i],2)
            plt.scatter(self.K2C,self.lnA_L.loc[i])
            if i%2==0:
                ls_b.append(b)
                ls_c.append(c)
        plt.show()
        return np.array(ls_c),np.array(ls_b)



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

        df_2theta_FWHM.to_csv("./data/modified_WA/2theta_FWHM.csv")
        df_thetas_intensity.to_csv("./data/modified_WA/fitting.csv",index=False)
        
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

    def H2(self):
        H2 = []
        for orientation in self.orientations:
            h,k,l = orientation
            h,k,l = int(h),int(k),int(l)
            h2 = ((h*k)**2+(h*l)**2+(k*l)**2)/(h**2+k**2+l**2)**2
            H2.append(h2)
        return np.array(H2)

    def alpha(self):
        x = self.H2
        coef = []
        for i in range(0,10**4):
            alpha = 2*i*10**(-7)
            y = (self.delta_K**2-alpha)/self.K**2
            now = np.corrcoef(x,y)[1][0]
            coef.append(now**2)
        """
        plt.plot(np.arange(0,10**4),coef)
        plt.show()
        """
        idx = np.argmax(coef)
        alpha = 2*idx*10**(-7)
        return alpha

    
    #def fourier(self):
if __name__ == "__main__":
    ins = Main()

# %%
