import numpy as np
import random
from matplotlib import pyplot as plt

class ARS():
    tag = "ARS"

    def __init__(self, f, fprima, xi=[-4,1,4], lb=-np.inf, ub=np.inf, use_lower=False, ns=50, **fargs):        
        self.lb = lb
        self.ub = ub
        self.f = f
        self.fprima = fprima
        self.fargs = fargs
        
        self.ns = 50
        self.x = np.array(xi)
        self.h = self.f(self.x, **self.fargs)
        self.hprime = self.fprima(self.x, **self.fargs)

        self.offset = np.amax(self.h)
        self.h = self.h-self.offset 

        if self.lb == -np.inf:
            if not (self.hprime[0] > 0):
                print(f"Derivata iniziale: {self.hprime[0]}")
                raise IOError('Per supporti illimitati a sinistra (-inf), la derivata iniziale deve essere positiva.')

        if self.ub == np.inf:
            if not (self.hprime[-1] < 0):
                print(f"Derivata finale: {self.hprime[-1]}")
                raise IOError('Per supporti illimitati a destra (+inf), la derivata finale deve essere negativa.')
        self.insert() 

        
    def draw(self, N):
        samples = np.zeros(N)
        n=0
        while n < N:
            [xt,i] = self.sample()
            ht = self.f(xt, **self.fargs)
            hprimet = self.fprima(xt, **self.fargs)
            ht = ht - self.offset
            ut = self.h[i] + (xt-self.x[i])*self.hprime[i]

            u = random.random()
            if u < np.exp(ht-ut):
                samples[n] = xt
                n +=1

            if self.u.__len__() < self.ns:
                self.insert([xt],[ht],[hprimet])
            
        return samples

    
    def insert(self,xnew=[],hnew=[],hprimenew=[]):
        if xnew.__len__() > 0:
            x = np.hstack([self.x,xnew])
            idx = np.argsort(x)
            self.x = x[idx]
            self.h = np.hstack([self.h, hnew])[idx]
            self.hprime = np.hstack([self.hprime, hprimenew])[idx]

        self.z = np.zeros(self.x.__len__()+1)
        self.z[1:-1] = (np.diff(self.h) - np.diff(self.x*self.hprime))/-np.diff(self.hprime) 
        
        self.z[0] = self.lb; self.z[-1] = self.ub
        N = self.h.__len__()
        self.u = self.hprime[[0]+list(range(N))]*(self.z-self.x[[0]+list(range(N))]) + self.h[[0]+list(range(N))]

        self.s = np.hstack([0, np.cumsum(np.diff(np.exp(self.u)) / (self.hprime))])
        self.cu = self.s[-1]


    def sample(self):
        u = random.random()
        i = np.nonzero(self.s/self.cu < u)[0][-1] 

        xt = self.x[i] + (-self.h[i] + np.log(self.hprime[i]*(self.cu*u - self.s[i]) + 
        np.exp(self.u[i]))) / self.hprime[i]

        return [xt,i]

    def plotHull(self):
        xpoints = self.z
        ypoints = np.exp(self.u) 
        plt.plot(xpoints,ypoints)
        plt.show()
        