


# did not use the global one as that one contains
# extra features that gives issue to this neuron


# class LSTM:
#     """
#     The structure of RNN is very similar to hidden Markov self. 
#     However, the main difference is with how parameters are calculated and constructed.
#     As LSTM Neuron has his unique parameter structure

#     Resources:
#     ----------
#         https://christinakouridi.blog/2019/06/20/vanilla-lstm-numpy/
#         https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359
#         https://medium.com/coinmonks/character-to-character-rnn-with-pytorchs-lstmcell-cd923a6d0e72

#     Parameter:
#     ----------
#     n_inputs: int
#         number of inputs
#     vocab_size: int
#         the length of the vocabulary set. 
#         We need this size to define the input size to the LSTM cell.
#     """
#     def __init__(self, n_inputs, vocab_size, activation_output="softmax", optimizer="adam"):
#         self.n_inputs= n_inputs
#         self.vocab_size = vocab_size
#         self.activation_output = act_functions[activation_output]()


#         self.state_W_optimizer  = copy(opt_functions[optimizer]())
#         self.forget_W_optimizer = copy(opt_functions[optimizer]())
#         self.input_W_optimizer  = copy(opt_functions[optimizer]())
#         self.output_W_optimizer = copy(opt_functions[optimizer]())

#         self.state_b_optimizer  = copy(opt_functions[optimizer]())
#         self.forget_b_optimizer = copy(opt_functions[optimizer]())
#         self.input_b_optimizer  = copy(opt_functions[optimizer]())
#         self.output_b_optimizer = copy(opt_functions[optimizer]())

#         # output
#         std_v = (1.0 / np.sqrt(self.vocab_size))
#         self.W = np.random.randn(self.vocab_size, self.n_inputs) * std_v
#         self.W_gradient  = np.zeros_like(self.W)
#         self.W_opt       = np.zeros_like(self.W)
#         self.W_opt2      = np.zeros_like(self.W)

#         self.b = np.zeros((self.vocab_size, 1))
#         self.b_gradient  = np.zeros_like(self.b)
#         self.b_opt       = np.zeros_like(self.b)
#         self.b_opt2      = np.zeros_like(self.b)

#         # random parameter values
#         std = (1.0 / np.sqrt(self.vocab_size + self.n_inputs))
#         W = np.random.randn(self.n_inputs, self.n_inputs + self.vocab_size) * std
#         b = np.ones((self.n_inputs, 1))

#         # parameters
#         self.state  = CellState(W, b)
#         self.forget = ForgetGate(W, b)
#         self.input  = InputGate(W, b)
#         self.output = OutputGate(W, b)
#         self.unit_parts = [self.forget, self.input, self.state, self.output]

#     def reset_gradients(self, value):
#         """
#         Resets gradients to zero before each backpropagation
#         """
#         for unit_part in self.unit_parts:
#             unit_part.W_gradient.fill(value)
#             unit_part.b_gradient.fill(value)
        
#         # main unit
#         self.W_gradient.fill(value)
#         self.b_gradient.fill(value)
    
#     def limit_gradients(self, limit):
#         """
#         Limits the magnitude of gradients to avoid exploding gradients
#         """
#         for unit_part in self.unit_parts:
#             np.clip(unit_part.W_gradient, -limit, limit, out=unit_part.W_gradient)
#             np.clip(unit_part.b_gradient, -limit, limit, out=unit_part.b_gradient)
        
#         # main unit
#         np.clip(self.W_gradient, -limit, limit, out=self.W_gradient)
#         np.clip(self.b_gradient, -limit, limit, out=self.b_gradient)

#     def optimization(self, batch_num, learning_rate, beta1, beta2):
#         self.batch_num = batch_num
#         self.learning_rate = learning_rate
#         self.beta_1 = beta1
#         self.beta_2 = beta2
        
#         # for unit_part in self.unit_parts:
#             # self._optimization(unit_part)
        
#         # self.output_W = self.output_W_optimizer.update(self.output_W, self.output_Wd)

#         # main unit
#         self._optimization(self)

#         # self.state_optimizer
#         # self.forget_optimizer
#         # self.input_optimizer
#         # self.output_optimizer
        
#         # self.state.W = self.state_W_optimizer.update(self.state.W, self.state.W_gradient)
#         # self.forget.W = self.forget_W_optimizer.update(self.forget.W, self.forget.W_gradient)
#         # self.input.W = self.input_W_optimizer.update(self.input.W, self.input.W_gradient)
#         # self.output.W = self.output_W_optimizer.update(self.output.W, self.output.W_gradient)

#         # self.state.b = self.state_b_optimizer.update(self.state.b, self.state.b_gradient)
#         # self.forget.b = self.forget_b_optimizer.update(self.forget.b, self.forget.b_gradient)
#         # self.input.b = self.input_b_optimizer.update(self.input.b, self.input.b_gradient)
#         # self.output.b = self.output_b_optimizer.update(self.output.b, self.output.b_gradient)

#     def _optimization(self, obj):
#         "Adam optimization used to update to a optimized result"
#         # optimization Weights
#         obj.W_opt  = obj.W_opt  * self.beta_1 + (1 - self.beta_1) * obj.W_gradient
#         obj.W_opt2 = obj.W_opt2 * self.beta_2 + (1 - self.beta_2) * obj.W_gradient ** 2

#         m_correlated = obj.W_opt  / (1 - self.beta_1 ** self.batch_num)
#         v_correlated = obj.W_opt2 / (1 - self.beta_2 ** self.batch_num)
#         obj.W -= self.learning_rate * m_correlated / (np.sqrt(v_correlated) + 1e-8)

#         # optimization Biases
#         obj.b_opt  = obj.b_opt  * self.beta_1 + (1 - self.beta_1) * obj.b_gradient
#         obj.b_opt2 = obj.b_opt2 * self.beta_2 + (1 - self.beta_2) * obj.b_gradient ** 2

#         m_correlated = obj.b_opt  / (1 - self.beta_1 ** self.batch_num)
#         v_correlated = obj.b_opt2 / (1 - self.beta_2 ** self.batch_num)
#         obj.b -= self.learning_rate * m_correlated / (np.sqrt(v_correlated) + 1e-8)

#     def forward(self, x, h_prev, c_prev):
#         # forget gate result
#         f = self.forget.forward(x, h_prev)

#         # input gate result + C'
#         c_hat, i = self.input.forward(x, h_prev, self.state)

#         # cell state result
#         c = self.state.forward(f, c_prev, i, c_hat)

#         # output gate result + h result
#         h, o = self.output.forward(x, c, h_prev)

#         # cell output
#         xc = np.row_stack((h_prev, x))
#         v = np.dot(self.W, h) + self.b

#         y_hat = self.activation_output(v)
#         return y_hat, v, h, o, c, c_hat, i, f, xc

#     def backward(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
#         dv = np.copy(y_hat)
#         dv[y] -= 1 # yhat - y
#         self.b_gradient += dv
#         self.W_gradient += np.dot(dv, h.T)

#         # gate output
#         dh = self.output.derivative(self.W.T, dv, dh_next, c, o, z)

#         # state cell
#         dc = self.state.derivative(i, o, c, dh, dc_next, c_bar, z)

#         # gate input
#         self.input.derivative(c_bar, dc, i, z)

#         # gate forget
#         self.forget.derivative(c_prev, dc, f, z)

#         dz = (
#             np.dot(self.forget.W.T, self.forget.b_gradient) + 
#             np.dot( self.input.W.T,  self.input.b_gradient) + 
#             np.dot( self.state.W.T,  self.state.b_gradient) + 
#             np.dot(self.output.W.T, self.output.b_gradient)
#         )

#         dh_prev = dz[:self.n_inputs, :]
#         dc_prev = f * dc
#         return dh_prev, dc_prev


# ''' ---------------------------------
# Utility Functions
# Simple utility functions, placed in here to make the main code easier to read
# --------------------------------- '''


# # Initialises a dictionary with the same keys as d but with zero-vector values
# def init_dict_like(d):
#     return {k: np.zeros_like(v) for k, v in d.iteritems()}


# # Normalises a vector
# def normalise(vec):
#     return vec / np.sum(vec)

# # Generates a one-hot-vector of length len, with the ith element 1
# def one_hot_vec(len, i):
#     vec = np.zeros((len, 1))
#     vec[i] = 1

#     return vec


# # TODO: possibly replace with hard sigmoid (faster)
# def sigmoid(x, D):
#     if not D:
#         return 1 / (1 + np.exp(- x))
#     else:
#         s = sigmoid(x, False)
#         return s - (s ** 2)


# def tanh(x, D):
#     if not D:
#         return np.tanh(x)
#     else:
#         return 1.0 - (np.tanh(x) ** 2)


# # Initialises the LSTM weight matrices
# def init_lstm_weights(X_DIM, Y_DIM, zeroed):
#     def layer():
#         if zeroed:
#             return np.zeros((X_DIM, Y_DIM))
#         else:
#             return np.random.random((X_DIM, Y_DIM)) * 0.01

#     return {
#         'i': layer(),
#         'f': layer(),
#         'o': layer(),
#         'g': layer()
#     }

# from sklearn.model_selection import train_test_split

# def data(n_epochs, batch_size, biggest_number):
#     X = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)
#     y = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)

#     for i in range(n_epochs):
#         # One-hot encoding of nominal values
#         start = np.random.randint(2, 7)
#         one_hot = np.zeros((batch_size, biggest_number))
#         one_hot[np.arange(batch_size), np.linspace(start, start*batch_size, num=batch_size, dtype=int)] = 1

#         X[i] = one_hot
#         y[i] = np.roll(X[i], -1, axis=0)
#     y[:, -1, 1] = 1

#     # return dataset
#     return train_test_split(X, y, test_size=0.4)
# X_train, X_test, y_train, y_test = data(n_epochs=3000, batch_size=10, biggest_number=61)

# # define model
# layers = [
#     RNN(n_units=10, input_shape=(10, 61))
# ]
# network = Network(layers=layers, loss="CrossEntropy")

# # train network
# history_loss = network.fit(X_train, y_train, n_epochs=500, batch_size=512)

# # network result 
# for i in range(5):
#     print(f"""
#         question = [{' '.join(np.argmax(X_test[i], axis=1).astype('str'))}]
#         answer   = [{' '.join((np.argmax(y_test, axis=2)[i]).astype('str'))}]
#         predict  = [{' '.join((np.argmax(network.predict(X_test), axis=2)[i]).astype('str'))}]
#     """)










# usefull resources:
# https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e
# http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf
# http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf
# https://www.aclweb.org/anthology/P14-1140.pdf
# https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf
# http://proceedings.mlr.press/v32/graves14.pdf
# https://www.analyticsvilayer.hidden_Wdya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/
# https://github.com/revsic/numpy-rnn/blob/master/RNN_numpy.ipynb
# https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6
# http://sebastianruder.com/optimizing-gradient-descent/index.html

# class LSTMCell(LayerBase):
#     def __init__(
#         self,
#         n_out,
#         act_fn="Tanh",
#         gate_fn="Sigmoid",
#         init="glorot_uniform",
#         optimizer=None,
#     ):
#         """
#         A single step of a long short-term memory (LSTM) RNN.
#         Notes
#         -----
#         Notation:
#         - ``Z[t]``  is the input to each of the gates at timestep `t`
#         - ``A[t]``  is the value of the hidden state at timestep `t`
#         - ``Cc[t]`` is the value of the *candidate* cell/memory state at timestep `t`
#         - ``C[t]``  is the value of the *final* cell/memory state at timestep `t`
#         - ``Gf[t]`` is the output of the forget gate at timestep `t`
#         - ``Gu[t]`` is the output of the update gate at timestep `t`
#         - ``Go[t]`` is the output of the output gate at timestep `t`
#         Equations::
#             Z[t]  = stack([A[t-1], X[t]])
#             Gf[t] = gate_fn(Wf @ Z[t] + bf)
#             Gu[t] = gate_fn(Wu @ Z[t] + bu)
#             Go[t] = gate_fn(Wo @ Z[t] + bo)
#             Cc[t] = act_fn(Wc @ Z[t] + bc)
#             C[t]  = Gf[t] * C[t-1] + Gu[t] * Cc[t]
#             A[t]  = Go[t] * act_fn(C[t])
#         where `@` indicates dot/matrix product, and '*' indicates elementwise
#         multiplication.
#         Parameters
#         ----------
#         n_out : int
#             The dimension of a single hidden state / output on a given timestep.
#         act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
#             The activation function for computing ``A[t]``. Default is
#             `'Tanh'`.
#         gate_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
#             The gate function for computing the update, forget, and output
#             gates. Default is `'Sigmoid'`.
#         init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
#             The weight initialization strategy. Default is `'glorot_uniform'`.
#         optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
#             The optimization strategy to use when performing gradient updates
#             within the :meth:`update` method.  If None, use the :class:`SGD
#             <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
#             parameters. Default is None.
#         """  # noqa: E501
#         super().__init__(optimizer)

#         self.init = init
#         self.n_in = None
#         self.n_out = n_out
#         self.n_timesteps = None
#         self.act_fn = ActivationInitializer(act_fn)()
#         self.gate_fn = ActivationInitializer(gate_fn)()
#         self.parameters = {
#             "Wf": None,
#             "Wu": None,
#             "Wc": None,
#             "Wo": None,
#             "bf": None,
#             "bu": None,
#             "bc": None,
#             "bo": None,
#         }
#         self.is_initialized = False

#     def _init_params(self):
#         self.X = []
#         init_weights_gate = WeightInitializer(str(self.gate_fn), mode=self.init)
#         init_weights_act = WeightInitializer(str(self.act_fn), mode=self.init)

#         Wf = init_weights_gate((self.n_in + self.n_out, self.n_out))
#         Wu = init_weights_gate((self.n_in + self.n_out, self.n_out))
#         Wc = init_weights_act((self.n_in + self.n_out, self.n_out))
#         Wo = init_weights_gate((self.n_in + self.n_out, self.n_out))

#         bf = np.zeros((1, self.n_out))
#         bu = np.zeros((1, self.n_out))
#         bc = np.zeros((1, self.n_out))
#         bo = np.zeros((1, self.n_out))

#         self.parameters = {
#             "Wf": Wf,
#             "Wu": Wu,
#             "Wc": Wc,
#             "Wo": Wo,
#             "bf": bf,
#             "bu": bu,
#             "bc": bc,
#             "bo": bo,
#         }

#         self.gradients = {
#             "Wf": np.zeros_like(Wf),
#             "Wu": np.zeros_like(Wu),
#             "Wc": np.zeros_like(Wc),
#             "Wo": np.zeros_like(Wo),
#             "bf": np.zeros_like(bf),
#             "bu": np.zeros_like(bu),
#             "bc": np.zeros_like(bc),
#             "bo": np.zeros_like(bo),
#         }

#         self.derived_variables = {
#             "C": [],
#             "A": [],
#             "Gf": [],
#             "Gu": [],
#             "Go": [],
#             "Gc": [],
#             "Cc": [],
#             "n_timesteps": 0,
#             "current_step": 0,
#             "dLdA_accumulator": None,
#             "dLdC_accumulator": None,
#         }

#         self.is_initialized = True

#     def _get_params(self):
#         Wf = self.parameters["Wf"]
#         Wu = self.parameters["Wu"]
#         Wc = self.parameters["Wc"]
#         Wo = self.parameters["Wo"]
#         bf = self.parameters["bf"]
#         bu = self.parameters["bu"]
#         bc = self.parameters["bc"]
#         bo = self.parameters["bo"]
#         return Wf, Wu, Wc, Wo, bf, bu, bc, bo

#     @property
#     def hyperparameters(self):
#         """Return a dictionary containing the layer hyperparameters."""
#         return {
#             "layer": "LSTMCell",
#             "init": self.init,
#             "n_in": self.n_in,
#             "n_out": self.n_out,
#             "act_fn": str(self.act_fn),
#             "gate_fn": str(self.gate_fn),
#             "optimizer": {
#                 "cache": self.optimizer.cache,
#                 "hyperparameters": self.optimizer.hyperparameters,
#             },
#         }

#     def forward(self, Xt):
#         """
#         Compute the layer output for a single timestep.
#         Parameters
#         ----------
#         Xt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
#             Input at timestep t consisting of `n_ex` examples each of
#             dimensionality `n_in`.
#         Returns
#         -------
#         At: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
#             The value of the hidden state at timestep `t` for each of the `n_ex`
#             examples.
#         Ct: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
#             The value of the cell/memory state at timestep `t` for each of the
#             `n_ex` examples.
#         """
#         if not self.is_initialized:
#             self.n_in = Xt.shape[1]
#             self._init_params()

#         Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

#         self.derived_variables["n_timesteps"] += 1
#         self.derived_variables["current_step"] += 1

#         if len(self.derived_variables["A"]) == 0:
#             n_ex, n_in = Xt.shape
#             init = np.zeros((n_ex, self.n_out))
#             self.derived_variables["A"].append(init)
#             self.derived_variables["C"].append(init)

#         A_prev = self.derived_variables["A"][-1]
#         C_prev = self.derived_variables["C"][-1]

#         # concatenate A_prev and Xt to create Zt
#         Zt = np.hstack([A_prev, Xt])

#         Gft = self.gate_fn(Zt @ Wf + bf)
#         Gut = self.gate_fn(Zt @ Wu + bu)
#         Got = self.gate_fn(Zt @ Wo + bo)
#         Cct = self.act_fn(Zt @ Wc + bc)
#         Ct = Gft * C_prev + Gut * Cct
#         At = Got * self.act_fn(Ct)

#         # bookkeeping
#         self.X.append(Xt)
#         self.derived_variables["A"].append(At)
#         self.derived_variables["C"].append(Ct)
#         self.derived_variables["Gf"].append(Gft)
#         self.derived_variables["Gu"].append(Gut)
#         self.derived_variables["Go"].append(Got)
#         self.derived_variables["Cc"].append(Cct)
#         return At, Ct

#     def backward(self, dLdAt):
#         """
#         Backprop for a single timestep.
#         Parameters
#         ----------
#         dLdAt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
#             The gradient of the loss wrt. the layer outputs (ie., hidden
#             states) at timestep `t`.
#         Returns
#         -------
#         dLdXt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
#             The gradient of the loss wrt. the layer inputs at timestep `t`.
#         """
#         assert self.trainable, "Layer is frozen"

#         Wf, Wu, Wc, Wo, bf, bu, bc, bo = self._get_params()

#         self.derived_variables["current_step"] -= 1
#         t = self.derived_variables["current_step"]

#         Got = self.derived_variables["Go"][t]
#         Gft = self.derived_variables["Gf"][t]
#         Gut = self.derived_variables["Gu"][t]
#         Cct = self.derived_variables["Cc"][t]
#         At = self.derived_variables["A"][t + 1]
#         Ct = self.derived_variables["C"][t + 1]
#         C_prev = self.derived_variables["C"][t]
#         A_prev = self.derived_variables["A"][t]

#         Xt = self.X[t]
#         Zt = np.hstack([A_prev, Xt])

#         dA_acc = self.derived_variables["dLdA_accumulator"]
#         dC_acc = self.derived_variables["dLdC_accumulator"]

#         # initialize accumulators
#         if dA_acc is None:
#             dA_acc = np.zeros_like(At)

#         if dC_acc is None:
#             dC_acc = np.zeros_like(Ct)

#         # Gradient calculations
#         # ---------------------

#         dA = dLdAt + dA_acc
#         dC = dC_acc + dA * Got * self.act_fn.grad(Ct)

#         # compute the input to the gate functions at timestep t
#         _Go = Zt @ Wo + bo
#         _Gf = Zt @ Wf + bf
#         _Gu = Zt @ Wu + bu
#         _Gc = Zt @ Wc + bc

#         # compute gradients wrt the *input* to each gate
#         dGot = dA * self.act_fn(Ct) * self.gate_fn.grad(_Go)
#         dCct = dC * Gut * self.act_fn.grad(_Gc)
#         dGut = dC * Cct * self.gate_fn.grad(_Gu)
#         dGft = dC * C_prev * self.gate_fn.grad(_Gf)

#         dZ = dGft @ Wf.T + dGut @ Wu.T + dCct @ Wc.T + dGot @ Wo.T
#         dXt = dZ[:, self.n_out :]

#         self.gradients["Wc"] += Zt.T @ dCct
#         self.gradients["Wu"] += Zt.T @ dGut
#         self.gradients["Wf"] += Zt.T @ dGft
#         self.gradients["Wo"] += Zt.T @ dGot
#         self.gradients["bo"] += dGot.sum(axis=0, keepdims=True)
#         self.gradients["bu"] += dGut.sum(axis=0, keepdims=True)
#         self.gradients["bf"] += dGft.sum(axis=0, keepdims=True)
#         self.gradients["bc"] += dCct.sum(axis=0, keepdims=True)

#         self.derived_variables["dLdA_accumulator"] = dZ[:, : self.n_out]
#         self.derived_variables["dLdC_accumulator"] = Gft * dC
#         return dXt

#     def flush_gradients(self):
#         """Erase all the layer's derived variables and gradients."""
#         assert self.trainable, "Layer is frozen"

#         self.X = []
#         for k, v in self.derived_variables.items():
#             self.derived_variables[k] = []

#         self.derived_variables["n_timesteps"] = 0
#         self.derived_variables["current_step"] = 0

#         # reset parameter gradients to 0
#         for k, v in self.parameters.items():
#             self.gradients[k] = np.zeros_like(v)

# class LSTM(LayerBase):
#     def __init__(
#         self,
#         n_out,
#         act_fn="Tanh",
#         gate_fn="Sigmoid",
#         init="glorot_uniform",
#         optimizer=None,
#     ):
#         """
#         A single long short-term memory (LSTM) RNN layer.
#         Parameters
#         ----------
#         n_out : int
#             The dimension of a single hidden state / output on a given timestep.
#         act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
#             The activation function for computing ``A[t]``. Default is `'Tanh'`.
#         gate_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
#             The gate function for computing the update, forget, and output
#             gates. Default is `'Sigmoid'`.
#         init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
#             The weight initialization strategy. Default is `'glorot_uniform'`.
#         optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
#             The optimization strategy to use when performing gradient updates
#             within the :meth:`update` method.  If None, use the :class:`SGD
#             <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
#             default parameters. Default is None.
#         """  # noqa: E501
#         super().__init__(optimizer)

#         self.init = init
#         self.n_in = None
#         self.n_out = n_out
#         self.n_timesteps = None
#         self.act_fn = ActivationInitializer(act_fn)()
#         self.gate_fn = ActivationInitializer(gate_fn)()
#         self.is_initialized = False

#     def _init_params(self):
#         self.cell = LSTMCell(
#             n_in=self.n_in,
#             n_out=self.n_out,
#             act_fn=self.act_fn,
#             gate_fn=self.gate_fn,
#             init=self.init,
#         )
#         self.is_initialized = True

#     @property
#     def hyperparameters(self):
#         """Return a dictionary containing the layer hyperparameters."""
#         return {
#             "layer": "LSTM",
#             "init": self.init,
#             "n_in": self.n_in,
#             "n_out": self.n_out,
#             "act_fn": str(self.act_fn),
#             "gate_fn": str(self.gate_fn),
#             "optimizer": self.cell.hyperparameters["optimizer"],
#         }

#     def forward(self, X):
#         """
#         Run a forward pass across all timesteps in the input.
#         Parameters
#         ----------
#         X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
#             Input consisting of `n_ex` examples each of dimensionality `n_in`
#             and extending for `n_t` timesteps.
#         Returns
#         -------
#         Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
#             The value of the hidden state for each of the `n_ex` examples
#             across each of the `n_t` timesteps.
#         """
#         if not self.is_initialized:
#             self.n_in = X.shape[1]
#             self._init_params()

#         Y = []
#         n_ex, n_in, n_t = X.shape
#         for t in range(n_t):
#             yt, _ = self.cell.forward(X[:, :, t])
#             Y.append(yt)
#         return np.dstack(Y)

#     def backward(self, dLdA):
#         """
#         Run a backward pass across all timesteps in the input.
#         Parameters
#         ----------
#         dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
#             The gradient of the loss with respect to the layer output for each
#             of the `n_ex` examples across all `n_t` timesteps.
#         Returns
#         -------
#         dLdX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_in`, `n_t`)
#             The value of the hidden state for each of the `n_ex` examples
#             across each of the `n_t` timesteps.
#         """  # noqa: E501
#         assert self.cell.trainable, "Layer is frozen"
#         dLdX = []
#         n_ex, n_out, n_t = dLdA.shape
#         for t in reversed(range(n_t)):
#             dLdXt, _ = self.cell.backward(dLdA[:, :, t])
#             dLdX.insert(0, dLdXt)
#         dLdX = np.dstack(dLdX)
#         return dLdX

#     @property
#     def derived_variables(self):
#         """
#         Return a dictionary containing any intermediate variables computed
#         during the forward / backward passes.
#         """
#         return self.cell.derived_variables

#     @property
#     def gradients(self):
#         """
#         Return a dictionary of the gradients computed during the backward
#         pass
#         """
#         return self.cell.gradients

#     @property
#     def parameters(self):
#         """Return a dictionary of the current layer parameters"""
#         return self.cell.parameters

#     def freeze(self):
#         """
#         Freeze the layer parameters at their current values so they can no
#         longer be updated.
#         """
#         self.cell.freeze()

#     def unfreeze(self):
#         """Unfreeze the layer parameters so they can be updated."""
#         self.cell.unfreeze()

#     def set_params(self, summary_dict):
#         """
#         Set the layer parameters from a dictionary of values.
#         Parameters
#         ----------
#         summary_dict : dict
#             A dictionary of layer parameters and hyperparameters. If a required
#             parameter or hyperparameter is not included within `summary_dict`,
#             this method will use the value in the current layer's
#             :meth:`summary` method.
#         Returns
#         -------
#         layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
#             The newly-initialized layer.
#         """
#         self = super().set_params(summary_dict)
#         return self.cell.set_parameters(summary_dict)

#     def flush_gradients(self):
#         """Erase all the layer's derived variables and gradients."""
#         self.cell.flush_gradients()

#     def update(self):
#         """
#         Update the layer parameters using the accrued gradients and layer
#         optimizer. Flush all gradients once the update is complete.
#         """
#         self.cell.update()
#         self.flush_gradients()




# # import numpy as np               #for maths
# # import pandas as pd              #for data manipulation
# # import matplotlib.pyplot as plt  #for visualization

# # import sys, os
# # sys.path.insert(1, os.getcwd() + "./../../network") 
# # from layers import LSTM
# # from algorithms.loss_functions import loss_functions

# # loss_function = loss_functions["CrossEntropy"]()


# # #data 
# # path = './NationalNames.csv'
# # data = pd.read_csv(path)

# # #get names from the dataset
# # data['Name'] = data['Name']

# # #get first 10000 names
# # data = np.array(data['Name'][:10000]).reshape(-1,1)

# # #covert the names to lowee case
# # data = [x.lower() for x in data[:,0]]

# # data = np.array(data).reshape(-1,1)

# # #to store the transform data
# # transform_data = np.copy(data)

# # #find the max length name
# # max_length = 0
# # for index in range(len(data)):
# #     max_length = max(max_length,len(data[index,0]))

# # #make every name of max length by adding '.'
# # for index in range(len(data)):
# #     length = (max_length - len(data[index,0]))
# #     string = '.'*length
# #     transform_data[index,0] = ''.join([transform_data[index,0],string])

# # print("Transformed Data")
# # print(transform_data[1:10])

# # #to store the vocabulary
# # vocab = list()
# # for name in transform_data[:,0]:
# #     vocab.extend(list(name))

# # vocab = set(vocab)
# # vocab_size = len(vocab)

# # print("Vocab size = {}".format(len(vocab)))
# # print("Vocab      = {}".format(vocab))

# # #map char to id and id to chars
# # char_id = dict()
# # id_char = dict()

# # for i,char in enumerate(vocab):
# #     char_id[char] = i
# #     id_char[i] = char

# # print('a-{}, 22-{}'.format(char_id['a'],id_char[22]))


# # # list of batches of size = 20
# # train_dataset = []

# # batch_size = 20

# # #split the trasnform data into batches of 20
# # for i in range(len(transform_data)-batch_size+1):
# #     start = i*batch_size
# #     end = start+batch_size
    
# #     #batch data
# #     batch_data = transform_data[start:end]
    
# #     if(len(batch_data)!=batch_size):
# #         break
        
# #     #convert each char of each name of batch data into one hot encoding
# #     char_list = []
# #     for k in range(len(batch_data[0][0])):
# #         batch_dataset = np.zeros([batch_size,len(vocab)])
# #         for j in range(batch_size):
# #             name = batch_data[j][0]
# #             char_index = char_id[name[k]]
# #             batch_dataset[j,char_index] = 1.0
     
# #         #store the ith char's one hot representation of each name in batch_data
# #         char_list.append(batch_dataset)
    
# #     #store each char's of every name in batch dataset into train_dataset
# #     train_dataset.append(char_list)

# # ########################################### Network ###########################################

# # #number of input units or embedding size
# # input_units = 100

# # #number of hidden neurons
# # hidden_units = 256

# # #number of output units i.e vocab size
# # output_units = vocab_size

# # #learning rate
# # learning_rate = 0.005

# # model = LSTM(n_units=input_units, input_shape=(hidden_units, output_units))

# # # loss function
# # def cal_loss_accuracy(batch_labels):
# #     loss = 0  #to sum loss for each time step
# #     prob = 1  #probability product of each time step predicted char
    
# #     #batch size
# #     batch_size = batch_labels[0].shape[0]
    
# #     #loop through each time step
# #     for i in range(1,len(model.state.outputs)+1):
# #         #get true labels and predictions
# #         labels = batch_labels[i]
# #         pred = model.state.outputs[i-1]
        
# #         prob = np.multiply(prob,np.sum(np.multiply(labels,pred),axis=1).reshape(-1,1))
# #         loss += np.sum((np.multiply(labels,np.log(pred)) + np.multiply(1-labels,np.log(1-pred))),axis=1).reshape(-1,1)
    

# #     # func = loss_functions["CrossEntropy"]()
# #     # y = batch_labels[:-1]
# #     # y_pred = list()
# #     # for key, output in model.state.outputs.items():
# #     #     y_pred.append(output)
# #     # values = func(np.array(y), np.array(y_pred))
    
# #     # loss = 0.00
# #     # for value in values:
# #     #     for value2 in value:
# #     #         for value3 in value2:
# #     #             loss += value3

# #     # # loss = np.sum(values[0][0])
# #     # # print("CHECK")
# #     # # print(np.mean(values[0][0]))
# #     # print(loss)
# #     # return 


# #     # calculate loss
# #     loss = np.sum(loss) * (-1 / batch_size)
# #     return loss

# # def calculate_output_cell_error(batch_labels):
# #     #to store the output errors for each time step
# #     output_error_cache = dict()
# #     activation_error_cache = dict()
# #     how = model.state.W
    
# #     #loop through each time step
# #     for i in range(1,len(model.state.outputs)+1):
# #         #get true and predicted labels
# #         labels = batch_labels[i]
# #         pred = model.state.outputs[i-1]
        
# #         #calculate the output_error for time step 't'
# #         error_output = pred - labels
        
# #         #calculate the activation error for time step 't'
# #         error_activation = np.matmul(error_output,how.T)
        
# #         #store the output and activation error in dict
# #         output_error_cache['eo'+str(i)] = error_output
# #         activation_error_cache['ea'+str(i)] = error_activation
        
# #     return output_error_cache,activation_error_cache

# # #calculate error for single lstm cell
# # def calculate_single_lstm_cell_error(activation_output_error,next_activation_error,next_cell_error, i):
# #     # activation error =  error coming from output cell and error coming from the next lstm cell
# #     activation_error = activation_output_error + next_activation_error
    
# #     # output gate error
# #     oa = model.output.outputs[i-1]
# #     eo = np.multiply(activation_error,model.activation(model.cell_cache[i-1]))
# #     eo = np.multiply(np.multiply(eo,oa),1-oa)
    
# #     # cell activation error
# #     cell_error = np.multiply(activation_error,oa)
# #     cell_error = np.multiply(cell_error,model.activation.derivative(model.activation(model.cell_cache[i-1])))
    
# #     # error also coming from next lstm cell 
# #     cell_error += next_cell_error
    
# #     # input gate error
# #     ia = model.input.outputs[i-1]
# #     ga = model.gate.outputs[i-1]
# #     ei = np.multiply(cell_error,ga)
# #     ei = np.multiply(np.multiply(ei,ia),1-ia)
    
# #     # gate gate error
# #     eg = np.multiply(cell_error,ia)
# #     eg = np.multiply(eg,model.activation.derivative(ga))
    
# #     # forget gate error
# #     fa = model.forget.outputs[i-1]
# #     ef = np.multiply(cell_error,model.cell_cache[i-2])
# #     ef = np.multiply(np.multiply(ef,fa),1-fa)
    
# #     # prev cell error
# #     prev_cell_error = np.multiply(cell_error,fa)
    
# #     # get parameters
# #     fgw = model.forget.W
# #     igw = model.input.W
# #     ggw = model.gate.W
# #     ogw = model.output.W
    
# #     # embedding + hidden activation error
# #     embed_activation_error = np.matmul(ef,fgw.T)
# #     embed_activation_error += np.matmul(ei,igw.T)
# #     embed_activation_error += np.matmul(eo,ogw.T)
# #     embed_activation_error += np.matmul(eg,ggw.T)
    
# #     input_hidden_units = fgw.shape[0]
# #     hidden_units = fgw.shape[1]
# #     input_units = input_hidden_units - hidden_units
    
# #     #prev activation error
# #     prev_activation_error = embed_activation_error[:,input_units:]
    
# #     #input error (embedding error)
# #     embed_error = embed_activation_error[:,:input_units]
    
# #     return prev_activation_error,prev_cell_error,embed_error

# # #calculate output cell derivatives
# # def calculate_output_cell_derivatives(output_error_cache,activation_cache):
# #     #to store the sum of derivatives from each time step
# #     dhow = np.zeros(model.state.W.shape)
    
# #     batch_size = activation_cache['a1'].shape[0]
    
# #     #loop through the time steps 
# #     for i in range(1,len(output_error_cache)+1):
# #         #get output error
# #         output_error = output_error_cache['eo' + str(i)]
        
# #         #get input activation
# #         activation = activation_cache['a'+str(i)]
        
# #         #cal derivative and summing up!
# #         dhow += np.matmul(activation.T,output_error)/batch_size
        
# #     return dhow

# # #backpropagation
# # def backward_propagation(loss_gradient):
# #     # to store embeding errors for each time step
# #     embedding_error_cache = dict()
    
# #     # next activation error next cell error  
# #     # for last cell will be zero
# #     eat = np.zeros(loss_gradient['ea1'].shape)
# #     ect = np.zeros(loss_gradient['ea1'].shape)
    
# #     #calculate all lstm cell errors (going from last time-step to the first time step)
# #     for i in range(len(model.state.outputs),0,-1):
# #         #calculate the lstm errors for this time step 't'
# #         pae,pce,ee = calculate_single_lstm_cell_error(loss_gradient['ea'+str(i)], eat, ect, i)
        
# #         #store the embedding error in dict
# #         embedding_error_cache['eemb'+str(i-1)] = ee
        
# #         #update the next activation error and next cell error for previous cell
# #         eat = pae
# #         ect = pce

# #     return embedding_error_cache

# # #update the Embeddings
# # def update_embeddings(embeddings,embedding_error_cache,batch_labels):
# #     #to store the embeddings derivatives
# #     embedding_derivatives = np.zeros(embeddings.shape)
    
# #     batch_size = batch_labels[0].shape[0]
    
# #     #sum the embedding derivatives for each time step
# #     for i in range(len(embedding_error_cache)):
# #         embedding_derivatives += np.matmul(batch_labels[i].T,embedding_error_cache['eemb'+str(i)])/batch_size
    
# #     #update the embeddings
# #     embeddings = embeddings - learning_rate*embedding_derivatives
# #     return embeddings

# # #train function
# # def fit(train_dataset,n_epochs=1000,batch_size=20):
    
# #     #generate the random embeddings
# #     embeddings = np.random.normal(0,0.01,(len(vocab),input_units))
# #     loss_history = []
    
# #     for epoch in range(n_epochs):
# #         #get batch dataset
# #         index = epoch%len(train_dataset)
# #         batches = train_dataset[index]
        
# #         #forward propagation
# #         embedding_cache = model.forward(batches,embeddings)
        
# #         # calculate the loss
# #         loss = cal_loss_accuracy(batches)
        
# #         # calculate output errors 
# #         _, loss_gradient = calculate_output_cell_error(batches)

# #         #backward propagation
# #         embedding_error_cache = backward_propagation(batches, loss_gradient)
        
# #         #update the parameters
# #         model.forget.update_weights()
# #         model.input.update_weights()
# #         model.output.update_weights()
# #         model.gate.update_weights()
# #         model.state.update_weights()
        
# #         #update the embeddings
# #         embeddings = update_embeddings(embeddings,embedding_error_cache,batches)
        
# #         # network result training
# #         loss_history.append(loss)
# #         print(f"\r[{epoch}/{n_epochs}] loss:{round(loss, 2)}", end="")
    
# #     return embeddings, loss_history

# # embeddings, loss_history = fit(train_dataset,n_epochs=8000)


# import numpy as np
# import matplotlib.pyplot as plt
# from IPython import display
# import signal
# from random import uniform
# plt.style.use('seaborn-white')

# import sys, os
# sys.path.insert(1, os.getcwd() + "./../../network") 
# from layers import LSTM
# from algorithms.loss_functions import loss_functions

# loss_function = loss_functions["CrossEntropy"]()

# #  data 
# data = open('wonderland.txt', 'r').read()
# chars = sorted(list(set(data)))
# char2idx = {w: i for i,w in enumerate(chars)}
# idx2char = {i: w for i,w in enumerate(chars)}

# n_epochs = 40
# batch_size = 128
# sequences_step = 5
# seq_length = 100

# n_hidden = 100
# sequence_size = 25
# learning_rate = 1e-1
# weight_sd = 0.1
# z_size = n_hidden + len(chars)

# # generate data
# def generate_dataset():
#     # cut the text in semi-redundant sequences of seq_length characters
#     sentences  = []
#     next_chars = []
#     for i in range(0, len(data) - sequence_size, sequences_step):
#         sentences.append(data[i : i+sequence_size])
#         next_chars.append(data[    i+sequence_size])

#     n_sentences = len(sentences)
#     n_chars = len(chars)
#     # print("Sentences size: ", n_sentences)

#     x = np.zeros((n_sentences, sequence_size, n_chars), dtype=np.int)
#     y = np.zeros((n_sentences, sequence_size, n_chars), dtype=np.int)
#     # y = np.zeros((n_sentences, n_chars), dtype=np.int)
#     for i, sentence in enumerate(sentences):
#         for t, char in enumerate(sentence):
#             x[i, t, char2idx[char]] = 1
#             if t > 0:
#                 y[i, t-1, char2idx[char]] = 1
#         y[i, -1, char2idx[next_chars[i]]] = 1
#     return x, y
# x, y = generate_dataset()

# model = LSTM(n_units=z_size, input_shape=(len(chars), n_hidden), activation='tanh', activation_output="softmax", optimizer="adagrad")

# def forward2(x, h_prev, C_prev, t):
#     assert x.shape == (len(chars), 1)
#     assert h_prev.shape == (n_hidden, 1)
#     assert C_prev.shape == (n_hidden, 1)
    
#     z = np.row_stack((h_prev, x))
#     model.forget.values[t] = model.sigmoid(np.dot(model.forget.W, z) + model.forget.b)
#     model.input.values[t] = model.sigmoid(np.dot(model.input.W, z) + model.input.b)
#     C_bar = model.activation(np.dot(model.cell.W, z) + model.cell.b)

#     C = model.forget.values[t] * C_prev + model.input.values[t] * C_bar
#     model.output.values[t] = model.sigmoid(np.dot(model.output.W, z) + model.output.b)
#     h = model.output.values[t] * model.activation(C)

#     v = np.dot(model.state.W, h) + model.state.b
#     y = model.activation_output(v)# np.exp(v) / np.sum(np.exp(v)) #softmax

#     return z, C_bar, C, model.output.values[t], h, v, y

# def fit(inputs, targets, x_train, y_train, h_prev, C_prev):
#     # Values at t - 1
#     model.state.values[-1] = np.copy(h_prev)
#     model.cell.values[-1] = np.copy(C_prev)

#     # one hot encode
#     x = np.zeros((len(inputs), len(chars), 1))
#     # x = np.zeros((len(inputs), len(chars)))
#     for t in range(len(inputs)):
#         x[t][inputs[t]] = 1
        
#     y = np.zeros((len(targets), len(chars)))
#     for t in range(len(targets)):
#         y[t][targets[t]] = 1

#     # # print(x[0])
#     # print(x.shape)
#     # print(x_train.shape)
#     # x = x_train[0]# same as x
#     # # print(x[0])

#     # # print(y[0])
#     # print(y.shape)
#     # print(y_train.shape)
#     # y = y_train[0]# same as y
#     # # print(y[0])

#     # Forward pass
#     loss = 0.0
#     y_pred = model.forward(x)

#     # print(y_pred)
#     # return 


#     loss_gradient = []
#     timesteps, _, _ = model.layer_input.shape
#     for t in range(timesteps):
#         loss += -np.log(y_pred[t][targets[t], 0])# loss
#         loss_gradient.append(np.copy(y_pred[t]))
#         loss_gradient[t][targets[t]] -= 1

#     # print(x)
#     # print(targets)
#     # print(x_train)
#     # print(y_train)

#     # print(x[0])
#     # print(x_train[0][0])


#     # print(loss_gradient)
#     # [[[ 0.01497192], [ 0.01140778], [ 0.01514817]]]
#     # return 


#     # Backward pass
#     loss_gradient = model.backward(loss_gradient)
        
#     return loss, model.state.values[len(inputs) - 1], model.cell.values[len(inputs) - 1]

# def sample(h_prev, C_prev, first_char_idx, sentence_length):
#     x = np.zeros((len(chars), 1))
#     x[first_char_idx] = 1

#     h = h_prev
#     C = C_prev

#     indexes = []
    
#     for t in range(sentence_length):
#         _, _, C, _, h, _, p = forward2(x, h, C, t)
#         idx = np.random.choice(range(len(chars)), p=p.ravel())
#         x = np.zeros((len(chars), 1))
#         x[idx] = 1
#         indexes.append(idx)

#     return indexes

# def update_status(inputs, h_prev, C_prev):
#     #initialized later
#     global plot_iter, plot_loss
#     global smooth_loss
    
#     # Get predictions for 200 letters with current model
#     sample_idx = sample(h_prev, C_prev, inputs[0], 200)
#     txt = ''.join(idx2char[idx] for idx in sample_idx)

#     #Print prediction and loss
#     print(f"----\n {txt} \n----")
#     print(f"iter {iteration}, loss: {smooth_loss}")

# # Exponential average of loss
# # Initialize to a error of a random model
# smooth_loss = -np.log(1.0 / len(chars)) * sequence_size

# iteration, pointer = 0, 0

# # For the graph
# plot_iter = np.zeros((0))
# plot_loss = np.zeros((0))

# while True:
#     if pointer + sequence_size >= len(data) or iteration == 0:
#         g_h_prev = np.zeros((n_hidden, 1))
#         g_C_prev = np.zeros((n_hidden, 1))
#         pointer = 0

#     inputs  = ([char2idx[ch] for ch in data[pointer: pointer + sequence_size]])
#     targets = ([char2idx[ch] for ch in data[pointer + 1: pointer + sequence_size + 1]])
#     loss, g_h_prev, g_C_prev = fit(inputs, targets, x, y, g_h_prev, g_C_prev)

#     smooth_loss = smooth_loss * 0.999 + loss * 0.001

#     # Print every hundred steps
#     if iteration % 100 == 0:
#         # , model.state.values[len(inputs) - 1], model.cell.values[len(inputs) - 1]
#         update_status(inputs, g_h_prev, g_C_prev)


#     plot_iter = np.append(plot_iter, [iteration])
#     plot_loss = np.append(plot_loss, [loss])

#     pointer += sequence_size
#     iteration += 1



# # # # https://github.com/pangolulu/rnn-from-scratch
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import sys, os, keras

# # # # from network.LSTM import LSTM
# # # import data
# # # sys.path.insert(1, os.getcwd() + "../../../network") 
# # # from layers import LSTM

# # # class Model:
# # #     def __init__(self, seq_length, seq_step, chars, char2idx, idx2char, n_neurons=100):
# # #         """
# # #         Implementation of simple character-level LSTM using Numpy
# # #         """
# # #         self.seq_length = seq_length # no. of time steps, also size of mini batch
# # #         self.seq_step   = seq_step   # no. size of each time step
# # #         self.vocab_size = len(chars) # no. of unique characters in the training data
# # #         self.char2idx   = char2idx   # characters to indices mapping
# # #         self.idx2char   = idx2char   # indices to characters mapping
# # #         self.n_neurons  = n_neurons  # no. of units in the hidden layer
        
# # #         self.unit = LSTM(self.n_neurons, self.vocab_size)
# # #         self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length

# # #     def sample(self, h_prev, c_prev, sample_size):
# # #         """
# # #         Outputs a sample sequence from the model
# # #         """
# # #         x = np.zeros((self.vocab_size, 1))
# # #         h = h_prev
# # #         c = c_prev
# # #         sample_string = ""

# # #         for t in range(sample_size):
# # #             y_hat, _, h, _, c, _, _, _, _ = self.unit.forward(x, h, c)

# # #             # get a random index within the probability distribution of y_hat(ravel())
# # #             idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
# # #             x = np.zeros((self.vocab_size, 1))
# # #             x[idx] = 1

# # #             # find the char with the sampled index and concat to the output string
# # #             char = self.idx2char[idx]
# # #             sample_string += char
# # #         return sample_string

# # #     def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
# # #         """
# # #         Implements the forward and backward propagation for one batch
# # #         """
# # #         x, z = {}, {}
# # #         f, i, c_bar, c, o = {}, {}, {}, {}, {}
# # #         y_hat, v, h = {}, {}, {}

# # #         # Values at t= - 1
# # #         h[-1] = h_prev
# # #         c[-1] = c_prev

# # #         loss = 0
# # #         for t in range(self.seq_length):
# # #             x[t] = np.zeros((self.vocab_size, 1))
# # #             x[t][x_batch[t]] = 1

# # #             y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = self.unit.forward(x[t], h[t - 1], c[t - 1])
# # #             loss += -np.log(y_hat[t][y_batch[t], 0])

# # #         self.unit.reset_gradients(0)
# # #         dh_next = np.zeros_like(h[0])
# # #         dc_next = np.zeros_like(c[0])

# # #         for t in reversed(range(self.seq_length)):
# # #             dh_next, dc_next = self.unit.backward(y_batch[t], y_hat[t], dh_next, dc_next, c[t - 1], z[t], f[t], i[t], c_bar[t], c[t], o[t], h[t])

# # #         return loss, h[self.seq_length - 1], c[self.seq_length - 1]

# # #     def train(self, X, y, epochs=10, learning_rate=0.01, beta1=0.9, beta2=0.999, verbose=True):
# # #         """
# # #         Main method of the LSTM class where training takes place
# # #         """
# # #         losses = [] # return history losses

# # #         for epoch in range(epochs):
# # #             h_prev = np.zeros((self.n_neurons, 1))
# # #             c_prev = np.zeros((self.n_neurons, 1))

# # #             for i in range(len(X)):
# # #                 x_batch = X[i] 
# # #                 y_batch = np.concatenate([ x_batch[1:], [y[i]] ])
                
# # #                 # Forward Pass
# # #                 loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

# # #                 # smooth out loss and store in list
# # #                 self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

# # #                 # keep loss history
# # #                 losses.append(self.smooth_loss)

# # #                 # overflowding protection
# # #                 self.unit.limit_gradients(5)

# # #                 batch_num = epoch * epochs + i / self.seq_length + 1
# # #                 self.unit.optimization(batch_num, learning_rate, beta1, beta2)

# # #                 # print out loss and sample string
# # #                 if verbose:
# # #                     if i % 100 == 0:
# # #                         prediction = self.sample(h_prev, c_prev, sample_size=250)

# # #                         print("-" * 100)
# # #                         print(f"Epoch:[{epoch}] Loss:{round(self.smooth_loss[0], 2)} Index:[{i}/{len(X)}]")
# # #                         print("-" * 88 + " prediction:")
# # #                         print(prediction + "\n")

# # #         return losses



# # # if __name__ == "__main__":
# # #     """
# # #     Implementation of simple character-level LSTM using Numpy
# # #     """
# # #     # get data
# # #     x, y = data.vectorization()
# # #     print(f'data has {len(data.text)} characters, {data.chars} are unique')

# # #     # define model
# # #     model = Model(data.seq_length, data.sequences_step, data.chars, data.char2idx, data.idx2char)

# # #     # train model
# # #     losses = model.train(x, y)

# # #     # display history losses
# # #     plt.plot([i for i in range(len(losses))], losses)
# # #     plt.xlabel("#training iterations")
# # #     plt.ylabel("training loss")
# # #     plt.show()

# # # # Print:
# # # # ----------------------------------------------------------------------------------------------------
# # # # Epoch:[0] Loss:461.54 Index:[0/14266]
# # # # ---------------------------------------------------------------------------------------- prediction:
# # # # vEкmS‘0.NJUTбђгyShsјOVTгф.uгTrAs?Cv(aKrXjCvfтN(јaђsowpCyRAT?вuCTrir5E2FђгH~FTSiveSLCN4CfтAycfTI31~gX9”%AzјnypуpбHTв4ELt"tSOу””rNZгf?CZhDqt

# # # # ......

# # # # ----------------------------------------------------------------------------------------------------
# # # # Epoch:[4] Loss:279.69 Index:[5700/14266]
# # # # ---------------------------------------------------------------------------------------- prediction:
# # # # a’no’r no ctu eunst. I tox gsga lrrntt s fy I’s oorsy, tegoe tam o euesools wr’mo ieoicos, bg auo I, slsm osooinnuetr’t guutom o m souuoand htth ii g uy o imoo goty’goo i tavrcamnuet y”ott tig uog uxctiuo sd ouru, taan H tQub teivwel “geeEuorss te “r

# # # # but, it is a little to slow :)





