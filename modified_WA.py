import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Main:
    def __init__(self):
        self.dir = "./data/fitting/output/predicts"

        self.orientations = ["200","220","310","222"] # remove 110 211
        self.crete_data()
        self.params = pd.read_csv("./data/modified_WA/2theta_FWHM.csv")
        self.fitting = pd.read_csv("./data/modified_WA/fitting.csv")

        self.lam = 0.070931
        self.K   = self.K()
        self.delta_K = self.delta_K()

        self.ch00 = 0.285
        self.H2    = self.H2()

        self.alpha = self.alpha()
        self.a,self.b = np.polyfit(self.H2,(self.delta_K**2-self.alpha)*10**4/self.K**2,1)
        self.q = -self.a/self.b
        print("q",self.q)
        self.C_bar = self.ch00*(1-self.q*self.H2)
        # こっから怪しい
        self.intensity = self.fitting["200intensity"]/self.fitting["200intensity"].max()
        plt.plot(range(len(self.intensity)),self.intensity)
        idx = np.argmax(self.intensity)
        wid = min(idx,(len(self.intensity)-1)-idx)
        self.intensity = self.intensity[idx-wid:idx+wid]
        plt.plot(range(len(self.intensity)),self.intensity)
        plt.show()
        fft = abs(np.fft.fft(self.intensity).real)
        plt.plot(np.arange(50),(fft/np.max(fft))[0:50])
        plt.show()
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
        theta = self.params["2theta"].to_numpy()/2
        theta = np.pi*theta/180
        K = 2*np.sin(theta)/self.lam
        return K
    
    def delta_K(self):
        theta = self.params["2theta"].to_numpy()/2
        theta = np.pi*theta/180
        delta_theta = self.params["FWHM"].to_numpy()
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
        plt.plot(np.arange(0,10**4),coef)
        plt.show()
        idx = np.argmax(coef)
        alpha = 2*idx*10**(-7)
        return alpha

    
    #def fourier(self):
if __name__ == "__main__":
    ins = Main()