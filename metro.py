
# Metropolis hastings for fitting a gaussian line

import numpy as np
import scipy
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))


# Lets create a gaussian line with some noise

N = 100. # nb points
x = np.arange(N) # axis

# parameters of the initial gaussian
sigma = 5.
moyenne = 50.
maxi = 1.

# parameter of the noise (here gaussian noise sampled relatively to the signal maximum).
noiselevel = maxi / 10.

# signal = gaussian g + noise
g = maxi  * gaussian(x, moyenne, sigma)
noise = noiselevel * np.random.normal(0,1,N)

signal = g + noise

# Take a 'random' fit parameters (should be real random but who cares) (also, it is obviously not a fit, it is just the model)
# In this example only sigma and the moyenne vary. The max is constant. To be changed.

sigma_fit = 2.5
moyenne_fit = 22.
maxi_fit = 1.00

fit = maxi_fit * gaussian(x, moyenne_fit, sigma_fit)


# Plots the signal and its random fit
plt.plot(x, signal)
plt.plot(x, fit)
#plt.show()
plt.savefig("signal.png")
plt.close()

lnL_old = - sum( (signal - fit) * (signal - fit) ) / (2. * noiselevel *
noiselevel)


#On varie juste en sigma et en moyenne pour le moment

Ntries = 15000
sigmas = []
lnLs = []
moyennes = []
old_sigma_fit = sigma_fit
old_moyenne_fit = moyenne_fit

sig_trajet = 0.5

for i in np.arange(Ntries) :
        new_sigma_fit = old_sigma_fit + (np.random.normal(0,sig_trajet,1))[0]
        new_moyenne_fit = old_moyenne_fit + (np.random.normal(0,sig_trajet,1))[0]

        new_fit = maxi_fit * gaussian(x, new_moyenne_fit, new_sigma_fit)

        lnL_new = - sum( (signal - new_fit) * (signal - new_fit) ) / (2. *
noiselevel * noiselevel)

        #print lnL_new
        #print lnL_old

        probab = np.exp(lnL_new - lnL_old) * 1.

        if  probab >=  np.random.uniform(0,1):
                old_moyenne_fit = new_moyenne_fit
                old_sigma_fit = new_sigma_fit
                lnL_old = lnL_new
                sigmas.append(old_sigma_fit)
                lnLs.append(lnL_old)
                moyennes.append(old_moyenne_fit)


# sigma et moyenne en fonction des Like
plt.plot(sigmas, lnLs, '+')
plt.plot(moyennes, lnLs, '+')
#plt.show()
plt.close()


# progression des sigma avec le temps
plt.plot(sigmas)
plt.show()
plt.close()




# progression des moyennes avec le temps
plt.plot(moyennes)
plt.show()
plt.close()


# Convergences moyennes et sigmas
plt.plot(sigmas, moyennes)
plt.show()
plt.close()



