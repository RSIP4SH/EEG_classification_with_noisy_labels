from keras.utils import to_categorical

class LatentSVDD:
    def __init__(self, nstates=7, size=10, seed=None):
        self.nstates = nstates
        self._used = False
        self.maxiter = 20

    def init_params(self):
        pass

    def phi(self, x):  # Phi
        pass
        return x

    def jointFeatureMap(self, x, latent):
        phi = self.phi(x)



    def fit(self, X, y):
        if not self._used:
            self.init_params()
            self._used = True
        psy = self.jointFeatureMap(X, z)

        pass