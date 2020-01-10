import numpy as np

class SubFunctions:
    def __init__(self,x,t):
        self.x,self.t = x,t
        
    def max_intensity(self):
        """
        Caluculate the peak of ka1
        the angle corresponding to the maxmun intensity
        """
        idx = np.argmax(self.t)
        return self.x[idx],self.t[idx]

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
        noise1 = np.mean(self.t[0:50])
        noise2 = np.mean(self.t[len(self.t)-51:len(self.t)-1])
        return (noise1+noise2)/2