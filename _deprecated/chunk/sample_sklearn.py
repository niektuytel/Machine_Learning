import numpy as np
import scipy.io
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

class Reservoir(object):
    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=None, connectivity=0.3, input_scaling=0.2, noise_level=0.01, circle=False):
        
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(n_internal_units, spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(n_internal_units, connectivity, spectral_radius)

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = spectral_radius
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = spectral_radius
                
        return internal_weights

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       

        return internal_weights

    def _compute_state_matrix(self, X, n_drop=0):
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units), dtype=float)
        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0 * np.random.binomial(1, 0.5 , [self._n_internal_units, V]) - 1.0) * self._input_scaling

        # compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop)
    
        # reservoir states on time reversed input
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states
    
    def getReservoirEmbedding(self, X,pca, ridge_embedding,  n_drop=5, bidir=True, test = False):

        res_states = self.get_states(X, n_drop=5, bidir=True)


        N_samples = res_states.shape[0]
        res_states = res_states.reshape(-1, res_states.shape[2])                   
        # ..transform..
        if test:
            red_states = pca.transform(res_states)
        else:
            red_states = pca.fit_transform(res_states)          
        # ..and put back in tensor form
        red_states = red_states.reshape(N_samples,-1,red_states.shape[1])  

        coeff_tr = []
        biases_tr = []   

        for i in range(X.shape[0]):
            ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeff_tr.append(ridge_embedding.coef_.ravel())
            biases_tr.append(ridge_embedding.intercept_.ravel())
        print(np.array(coeff_tr).shape,np.array(biases_tr).shape)
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr


data = scipy.io.loadmat('dataset/JpVow.mat')
X = data['X']  
Y = data['Y']
Xte = data['Xte']
Yte = data['Yte']

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
Y = onehot_encoder.fit_transform(Y)
Yte = onehot_encoder.transform(Yte)

pca = PCA(n_components=75)  
ridge_embedding = Ridge(alpha=10, fit_intercept=True)
readout = Ridge(alpha=5)

res = Reservoir(n_internal_units=450, spectral_radius=0.6, leak=0.6, connectivity=0.25, input_scaling=0.1, noise_level=0.01, circle=False)

input_repr = res.getReservoirEmbedding(X,pca, ridge_embedding,  n_drop=5, bidir=True, test=False)
input_repr_te = res.getReservoirEmbedding(Xte,pca, ridge_embedding,  n_drop=5, bidir=True, test=True)

readout.fit(input_repr, Y)
logits = readout.predict(input_repr_te)
pred_class = np.argmax(logits, axis=1)

# output
true_class = np.argmax(Yte, axis=1)
print(accuracy_score(true_class, pred_class))
print(f1_score(true_class, pred_class, average='weighted'))



