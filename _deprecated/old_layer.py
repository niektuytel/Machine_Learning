


























                

    # def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
    #     history_loss = []

    #     for epoch in range(n_epochs):
    #         batch_error = []
    #         # for X_batch, y_batch in self._gen_batch(X, y, batch_size):
    #         y_pred = self._forward(X)

    #         # the loss gradient with respect to y_pred
    #         gradient_loss = self.loss_function.gradient(y, y_pred)

    #         # Backpropagate. Update weights
    #         self._backward(gradient_loss, y_pred)

    #         batch_error.append(
    #             np.mean(self.loss_function(y, y_pred))
    #         )
            
    #         history_loss.append(
    #             np.mean(batch_error)
    #         )

    #         # # display trained network state
    #         # # print("called")
    #         # if verbose:
    #         #     print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

    #     # # model accuracy
    #     # if verbose:
    #     #     y_true = np.argmax(y, axis=2)
    #     #     y_pred = np.argmax(self.predict(X), axis=2)
    #     #     accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
    #     #     print(f"\nAccuracy: {accuracy}%\n")

    #     return history_loss

# # new design
# class Network2():
#     def __init__(self, layers=[], loss="MSE", activation_output=None):
#         self.layers = layers
#         self.loss_function = loss_functions[loss]()
#         self.activation_output = None

#         # activation on last layer
#         if activation_output != None:
#             self.activation_output = act_functions[activation_output]()

#         # set all layer input params
#         for i in range(len(self.layers)):
#             prev = min(0, i-1)
#             input_shape = (self.layers[prev].n_units,)
#             self.layers[i](input_shape)

#     def add(self, layer):
#         prev = min(0, len(self.layers)-1)

#         # set all layer input params
#         self.layers.append(layer)
#         input_shape = (self.layers[prev].n_units,)
#         self.layers[-1](input_shape)

#     def remove(self, layer_index):
#         del self.layers[layer_index]
#         return self.layers
    
#     def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
#         history_loss = []

#         for epoch in range(n_epochs):
#             batch_error = []
#             # for X_batch, y_batch in self._gen_batch(X, y, batch_size):
#             y_pred = self._forward(X)

#             # the loss gradient with respect to y_pred
#             gradient_loss = self.loss_function.gradient(y, y_pred)

#             # Backpropagate. Update weights
#             self._backward(gradient_loss, y_pred)

#             batch_error.append(
#                 np.mean(self.loss_function(y, y_pred))
#             )
            
#             history_loss.append(
#                 np.mean(batch_error)
#             )

#             # display trained network state
#             # print("called")
#             if verbose:
#                 print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

#         # model accuracy
#         if verbose:
#             y_true = np.argmax(y, axis=2)
#             y_pred = np.argmax(self.predict(X), axis=2)
#             accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
#             print(f"\nAccuracy: {accuracy}%\n")

#         return history_loss

#     def predict(self, X):
#         return self._forward(X)

#     def _backward(self, gradient_loss, y_pred):
#         # derive activation on output (softmax)
#         if self.activation_output != None:
#             gradient_loss = gradient_loss * self.activation_output.gradient(y_pred)

#         # update weights in each layer, bind the output to next input gradient loss
#         for layer in reversed(self.layers):
#             gradient_loss = layer.backward(gradient_loss)
            
#     def _forward(self, X):
#         output = X

#         # walk through layers, bind the output to input
#         for layer in self.layers:
#             output = layer.forward(output)

#         # activation on output (softmax)
#         if self.activation_output != None:
#             output = self.activation_output(output)

#         return output
        
#     def _gen_batch(self, X, y=None, batch_size=64):
#         n_samples = X.shape[0]
#         for i in np.arange(0, n_samples, batch_size):
#             begin, end = i, min(i+batch_size, n_samples)
#             if y is not None:
#                 yield X[begin:end], y[begin:end]
#             else:
#                 yield X[begin:end]

# class Layer(object):
#     def __init__(self, n_units, input_shape, activation, optimizer):
#         self.n_units      = n_units
#         self.input_shape  = input_shape
#         self.output_shape = (n_units,)
        
#         self.layer_input   = None
#         self.activation    = act_functions[activation]()
#         self.optimizer     = opt_functions[optimizer]()

# class Gate():
#     # Single gate used in recurrent neural networks
#     def __init__(self, optimizer, clip_limit=1):
#         # timestep values
#         self.clip_limit = clip_limit
#         self.values = []

#         # weights
#         self.W = None
#         self.W_gradient = None
#         self.W_optimizer = copy(optimizer)

#         # inputs
#         self.U = None
#         self.U_gradient = None
#         self.U_optimizer = copy(optimizer)

#         # bias
#         self.b = None
#         self.b_gradient = None
#         self.b_optimizer = copy(optimizer)

#     def initialize(self, shape):
#         # initialize Weight
#         limit = 1 / np.sqrt(shape[1])
        
#         self.W = np.random.uniform(-limit, limit, shape)
#         self.U = np.random.uniform(-limit, limit, (shape[0], shape[0]))
#         self.b = np.zeros((shape[0], 1))

#         self.W_gradient = np.zeros_like(self.W)
#         self.U_gradient = np.zeros_like(self.U)
#         self.b_gradient = np.zeros_like(self.b)

#     def clear_gradients(self):
#         self.W_gradient = np.zeros_like(self.W)
#         self.U_gradient = np.zeros_like(self.U)
#         self.b_gradient = np.zeros_like(self.b)

#     def update_weights(self):
#         # np.clip(self.W_gradient, -self.clip_limit, self.clip_limit, out=self.W_gradient)
#         # np.clip(self.b_gradient, -self.clip_limit, self.clip_limit, out=self.b_gradient)

#         self.W = self.W_optimizer.update(self.W, self.W_gradient)
#         self.U = self.U_optimizer.update(self.U, self.U_gradient)
#         self.b = self.b_optimizer.update(self.b, self.b_gradient)

# class Dense(Layer):
#     """
#         ## Params:
#         ```txt
#             n_units: int
#                 Number of neurons in a layer\n
#             input_shape: (int, int)
#                 Amount of input params on each neuron\n
#             activation: string
#                 Function name used to set neuron inputs between(e.g. [0 - 1])\n
#             optimizer: string
#                 Function name used to update params on back propergation\n
#             static_weights: bool
#                 Set default static weights.
#                 Keep the out coming results the same.\n
#         ```

#         ## Usable:
#         ```python
#             Dense(n_units=3)
#             Dense(n_units=3)(input_shape=(3,))
#             Dense(n_units=3, input_shape=(3,), activation="relu", optimizer="adam", static_weights=True)
#         ```

#         ### Sample:
#         ```python
#         from layers import *
#         X = np.array([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0,0,1],[0,1,1]]) # train data
#         Y = np.array([  [0],    [0],    [0],    [1],    [1],    [1]  ]) # train targets
        
#         # define model
#         network = Network(loss="MSE")
#         network.add(Dense(n_units=3, input_shape=(3,)))
#         network.add(Dense(n_units=1))
        
#         # train network
#         history_loss = network.fit(X,Y,20000)
            
#         # network result
#         print(f"question:[0, 1, 1] --- predict:{network.predict(np.array([0, 1, 1]))} --- target:1")
#         print(f"question:[1, 1, 0] --- predict:{network.predict(np.array([1, 1, 0]))} --- target:0")
#         print(f"question:[1, 0, 1] --- predict:{network.predict(np.array([1, 0, 1]))}")
#         ```
#     """
#     def __init__(self, n_units, input_shape=None, activation="relu", optimizer="adam", static_weights=True):
#         super().__init__(n_units, input_shape, activation, optimizer)
#         self.static_weights = static_weights

#         # optimizers for layer params
#         self.W_optimizer = copy(self.optimizer)
#         self.b_optimizer = copy(self.optimizer)
        
#         if self.input_shape != None:
#             self.__call__(self.input_shape)

#     def __call__(self, input_shape=None):
#         # input shape from previous layer
#         if self.input_shape is None:
#             self.input_shape = input_shape

#         # initialize weights
#         shape = (self.input_shape[0], self.n_units)
#         self.states_W = default_weights(shape, self.static_weights, 1)
#         self.b = np.zeros((1, self.n_units))

#     def forward(self, X):
#         # need for backprop
#         self.layer_input = X

#         # pass trough neuron
#         self.wsum = self.layer_input.dot(self.states_W) + self.b
#         return self.activation(self.wsum)
        
#     def backward(self, loss):
#         # backward gradient (chain-rule)
#         gradient = loss * self.activation.gradient(self.wsum)
#         gradient_W = self.layer_input.T.dot(gradient)
#         gradient_b = np.sum(gradient, axis=0, keepdims=True)

#         # Update the layer params
#         old_W  = self.states_W.T
#         self.states_W = self.W_optimizer.update(self.states_W, gradient_W)
#         self.b = self.b_optimizer.update(self.b, gradient_b)

#         # Return gradient for next layer
#         # Calculated based on the weights used during the forward pass
#         loss = gradient.dot(old_W)
#         return loss

# class RNN(Layer):
#     '''
#         ## Params:
#         ```txt
#             n_units: int
#                 Number of neurons in a layer\n
#             input_shape: (int, int)
#                 First value is the input size on each neuron,  
#                 the second value is the batch size of 1 input value.\n
#             activation: string
#                 Function name used to interact on given inputs/states, inside the neuron\n
#             activation_output: string
#                 Function name used to interact on given outputs, outputs in the neuron\n
#             optimizer: string
#                 Function name used to update params on back propergation\n
#             bptt_trunc: int
#                 Decides how many time steps the gradient should be propagated backwards  
#                 through states given the loss gradient for time step t.\n
#         ```

#         ## Usable:
#         ```python
#             RNN(n_units=10)
#             RNN(n_units=10)(input_shape=(3,))
#             RNN(n_units=10, input_shape=(3,50), activation="relu", activation_output="softmax", optimizer="adam", bptt_trunc=5)
#         ```

#         ## Sample:
#         ```python
#         import numpy as np
#         from layers import *
#         from sklearn.model_selection import train_test_split

#         def data(n_epochs, batch_size, biggest_number):
#             X = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)
#             y = np.zeros([n_epochs, batch_size, biggest_number], dtype=float)

#             for i in range(n_epochs):
#                 # One-hot encoding of nominal values
#                 start = np.random.randint(2, 7)
#                 one_hot = np.zeros((batch_size, biggest_number))
#                 one_hot[np.arange(batch_size), np.linspace(start, start*batch_size, num=batch_size, dtype=int)] = 1

#                 X[i] = one_hot
#                 y[i] = np.roll(X[i], -1, axis=0)
#             y[:, -1, 1] = 1

#             # return dataset
#             return train_test_split(X, y, test_size=0.4)
#         X_train, X_test, y_train, y_test = data(n_epochs=3000, batch_size=10, biggest_number=61)

#         # define model
#         layers = [
#             RNN(n_units=10, input_shape=(10, 61))
#         ]
#         network = Network(layers=layers, loss="CrossEntropy")

#         # train network
#         history_loss = network.fit(X_train, y_train, n_epochs=500, batch_size=512)

#         # network result 
#         for i in range(5):
#             print(f"""
#                 question = [{' '.join(np.argmax(X_test[i], axis=1).astype('str'))}]
#                 answer   = [{' '.join((np.argmax(y_test, axis=2)[i]).astype('str'))}]
#                 predict  = [{' '.join((np.argmax(network.predict(X_test), axis=2)[i]).astype('str'))}]
#             """)
#         ```
#     '''
#     def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam", bptt_trunc=5):
#         super().__init__(n_units, input_shape, activation, optimizer)
#         self.activation_output = act_functions[activation_output]()
#         self.bptt_trunc = bptt_trunc

#         # input
#         self.inputs = None
#         self.input_W = None
#         self.input_Wd = None
#         self.input_W_optimizer = copy(self.optimizer)
        
#         # states
#         self.states = None
#         self.states_W = None
#         self.states_Wd = None
#         self.states_W_optimizer = copy(self.optimizer)

#         # output
#         self.outputs = None
#         self.output_W = None
#         self.output_Wd = None
#         self.output_W_optimizer = copy(self.optimizer)

#         if self.input_shape != None:
#             self.__call__(self.input_shape)

#     def __call__(self, input_shape=None):
#         if self.input_shape is None:
#             self.input_shape = input_shape
    
#         timesteps, input_dim = self.input_shape
#         limit = 1 / np.sqrt(input_dim)

#         # initialize input Weight
#         shape_input  = (self.n_units, input_dim)
#         self.input_W = np.random.uniform(-limit, limit, shape_input)
        
#         limit = 1 / np.sqrt(self.n_units)

#         # initialize states Weight
#         shape_state  = (self.n_units, self.n_units)
#         self.states_W = np.random.uniform(-limit, limit, shape_state)

#         # initialize output Weight
#         shape_output = (input_dim, self.n_units)
#         self.output_W = np.random.uniform(-limit, limit, shape_output)


#     def forward(self, X):
#         return self.forward_pass(X)

#     def forward_pass(self, X):
#         # need for backprop
#         self.layer_input = X

#         batch_size, timesteps, input_dim = X.shape
#         self.inputs = np.zeros((batch_size, timesteps, self.n_units))
#         self.states = np.zeros((batch_size, timesteps+1, self.n_units))
#         self.outputs = np.zeros((batch_size, timesteps, input_dim))
        
#         # pass trough neuron
#         for t in range(timesteps):
#             self.inputs[:, t] = X[:, t].dot(self.input_W.T) + self.states[:, t-1].dot(self.states_W.T)
#             self.states[:, t] = self.activation(self.inputs[:, t])
#             self.outputs[:, t] = self.states[:, t].dot(self.output_W.T)

#         return self.activation_output(self.outputs)

#     def backward(self, loss):
#         return self.backward_pass(loss)

#     def backward_pass(self, loss):
#         # gradients
#         self.input_Wd  = np.zeros_like(self.input_W)
#         self.states_Wd = np.zeros_like(self.states_W)
#         self.output_Wd = np.zeros_like(self.output_W)

#         # backward gradient (chain-rule)
#         gradient = loss * self.activation_output.gradient(self.outputs)
#         gradient_next = np.zeros_like(gradient)

#         # update layer weights/params
#         _, timesteps, _ = gradient.shape
#         for t in reversed(range(timesteps)):
#             gradient_state = gradient[:, t].dot(self.output_W) * self.activation.gradient(self.inputs[:, t])
#             gradient_next[:, t] = gradient_state.dot(self.input_W)

#             self.output_Wd += gradient[:, t].T.dot(self.states[:, t])
#             for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
#                 self.input_Wd += gradient_state.T.dot(self.layer_input[:, t_])
#                 self.states_Wd += gradient_state.T.dot(self.states[:, t_-1])
                
#                 gradient_state = gradient_state.dot(self.states_W) * self.activation.gradient(self.inputs[:, t_-1])

#         self.input_W  = self.input_W_optimizer.update(self.input_W, self.input_Wd)
#         self.states_W = self.states_W_optimizer.update(self.states_W, self.states_Wd)
#         self.output_W = self.output_W_optimizer.update(self.output_W, self.output_Wd)

#         # Return gradient for next layer
#         # Calculated based on the weights used during the forward pass
#         return gradient_next

# # TODO: fix Back Propergation
# class LSTM(Layer):
#     def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam"):
#         super().__init__(n_units, input_shape, activation, optimizer)
#         self.sigmoid = act_functions["sigmoid"]()
#         self.activation_output = act_functions[activation_output]()

#         self.forget = Gate(self.optimizer)
#         self.input  = Gate(self.optimizer)
#         self.cell   = Gate(self.optimizer)
#         self.output = Gate(self.optimizer)
#         self.state = Gate(self.optimizer) 

#         if self.input_shape != None:
#             self.__call__(self.input_shape)

#     def __call__(self, input_shape=None):
#         if self.input_shape is None:
#             self.input_shape = input_shape
        
#         timesteps, input_dim = self.input_shape
#         shape_input  = (self.n_units, input_dim)
#         shape_state  = (self.n_units, self.n_units)
#         shape_output = (input_dim, self.n_units)
        
#         # initialize gates
#         self.forget.initialize(shape=shape_input) 
#         self.input.initialize(shape=shape_input)  
#         self.cell.initialize(shape=shape_input)   
#         self.output.initialize(shape=shape_input) 
#         self.state.initialize(shape=shape_input) 

#     def forward(self, X):
#         batch_size, timesteps, input_dim = X.shape
#         self.layer_input = X

#         self.forget.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.input.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.output.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.cell.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.state.values = np.zeros((batch_size, timesteps, self.n_units))

#         outputs = np.zeros((batch_size, timesteps, input_dim))

#         # forward pass
#         for t in range(timesteps):
#             Wf, Wi, Wo, Wc = self.forget.W, self.input.W, self.output.W, self.cell.W
#             Uf, Ui, Uo, Uc = self.forget.U, self.input.U, self.output.U, self.cell.U
#             xt = X[:, t]
#             ct_prev = self.cell.values[:, t-1]

#             ft = self.sigmoid(xt.dot(Wf.T) + ct_prev.dot(Uf))
#             it = self.sigmoid(xt.dot(Wi.T) + ct_prev.dot(Ui))
#             ot = self.sigmoid(xt.dot(Wo.T) + ct_prev.dot(Uo))
#             ct_hat = self.activation(xt.dot(Wc.T) + ct_prev.dot(Uc))# g
#             ct = ft * ct_prev + it * ct_hat
#             st = ot * self.activation(ct)#(ht)/(at)

#             self.forget.values[:, t] = ft
#             self.input.values[:, t] = it
#             self.output.values[:, t] = ot 
#             self.cell.values[:, t] = ct
#             self.state.values[:, t] = st#(ht)/(at)

#             # next ht
#             outputs[:, t] = self.state.values[:, t].dot(self.state.W)

#         return outputs

#     def backward(self, gradient):
#         print("TODO: fix Back Propergation")
#         gradient_next = np.zeros_like(gradient)

#         # gradients
#         self.forget.W_gradient = np.zeros_like(self.forget.W)
#         self.input.W_gradient = np.zeros_like(self.input.W)
#         self.output.W_gradient = np.zeros_like(self.output.W)
#         self.cell.W_gradient = np.zeros_like(self.cell.W)
#         self.state.W_gradient = np.zeros_like(self.state.W)

#         # update layer weights/params
#         _, timesteps, _ = gradient.shape
#         for t in reversed(range(timesteps)):
#             # TODO: fix Back Propergation
#             self.forget.W_gradient += 0.00
#             self.input.W_gradient  += 0.00
#             self.output.W_gradient += 0.00
#             self.cell.W_gradient   += 0.00
#             self.state.W_gradient  += 0.00

#         self.forget.update_weights()
#         self.input.update_weights()
#         self.output.update_weights()
#         self.cell.update_weights()
#         self.state.update_weights()

#         return gradient_next
    
# # TODO: fix Back Propergation
# class GRU(Layer):
#     def __init__(self, n_units, input_shape=None, activation='tanh', activation_output="softmax", optimizer="adam"):
#         super().__init__(n_units, input_shape, activation, optimizer)
#         self.sigmoid = act_functions["sigmoid"]()
#         self.activation_output = act_functions[activation_output]()


#         self.reset = Gate(self.optimizer)# rt
#         self.update = Gate(self.optimizer)# zt
#         self.hidden = Gate(self.optimizer)# ht

        


#         self.input    = Gate(self.optimizer)
#         self.output   = Gate(self.optimizer)
#         self.state    = Gate(self.optimizer)

#         if self.input_shape != None:
#             self.__call__(self.input_shape)

#     def __call__(self, input_shape=None):
#         if self.input_shape is None:
#             self.input_shape = input_shape
    
#         timesteps, input_dim = self.input_shape
#         shape_input  = (self.n_units, input_dim)
#         # shape_state  = (self.n_units, input_dim)
#         shape_state  = (self.n_units, self.n_units)
#         shape_output = (input_dim, self.n_units)
        
#         # initialize gates
#         self.reset.initialize(shape=shape_input)
#         self.update.initialize(shape=shape_input)
#         self.hidden.initialize(shape=shape_input)

#         # self.input.initialize(shape=shape_input)
#         self.output.initialize(shape=shape_output)
#         self.state.initialize(shape=shape_state)

#     def forward(self, X):
#         # need for backprop
#         self.layer_input = X

#         batch_size, timesteps, input_dim = X.shape
#         self.reset.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.update.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.hidden.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.state.values = np.zeros((batch_size, timesteps, self.n_units))
#         self.output.values = np.zeros((batch_size, timesteps, input_dim))
        
#         # pass trough neuron
#         for t in range(timesteps):
#             Wu, Wr, Wo, Wh = self.update.W, self.reset.W, self.output.W, self.hidden.W
#             Uu, Ur, Uo, Uh = self.update.U, self.reset.U, self.output.U, self.hidden.U
#             xt = X[:, t]
#             ht_prev = self.hidden.values[:, t-1]

#             zt = self.sigmoid(xt.dot(Wu.T) + ht_prev.dot(Uu))
#             rt = self.sigmoid(xt.dot(Wr.T) + ht_prev.dot(Ur))
#             ht_hat = self.activation(xt.dot(Wh.T) + np.dot((rt * ht_prev), Uh))
#             ht = ((1 - zt) * ht_prev) + (zt * ht_hat)
#             ot = ht.dot(Wh)

#             self.update.values[:, t] = zt
#             self.reset.values[:, t]  = rt
#             self.hidden.values[:, t] = ht
#             self.output.values[:, t] = ot
            
#         return self.output.values

#     def backward(self, gradient):
#         print("TODO: fix Back Propergation")
#         gradient_next = np.zeros_like(gradient)

#         # gradients
#         self.update.W_gradient = np.zeros_like(self.update.W)
#         self.reset.W_gradient = np.zeros_like(self.reset.W)
#         self.hidden.W_gradient = np.zeros_like(self.hidden.W)
#         self.state.W_gradient = np.zeros_like(self.state.W)
#         self.output.W_gradient = np.zeros_like(self.output.W)

#         # update layer weights/params
#         _, timesteps, _ = gradient.shape
#         for t in reversed(range(timesteps)):
#             # TODO: fix Back Propergation
#             self.update.W_gradient += 0.00
#             self.reset.W_gradient  += 0.00
#             self.hidden.W_gradient += 0.00
#             self.state.W_gradient   += 0.00
#             self.output.W_gradient  += 0.00

#         self.update.update_weights()
#         self.reset.update_weights()
#         self.hidden.update_weights()
#         self.state.update_weights()
#         self.output.update_weights()

#         return gradient_next


# Default Structure of layers
# class Input()
# class Output()
# class RNN(Layer):
#     def __init__(self, n_units, input_shape=None, activation="tanh", optimizer="adam"):
#         super().__init__(n_units, input_shape, activation, optimizer)
# 
#     def __call__(self, input_shape, static_weights=True):
#         # input shape from previous layer
#         if self.input_shape is None:
#             self.input_shape = input_shape
# 
#     def forward(self, X):
#         return X
#   
#     def backward(self, loss):
#         return loss

# class Network():
#     def __init__(self, layers=[], loss_name="MSE"):
#         self.layers = layers
#         self.loss_function = loss_functions[loss_name]()

#         # set all layer input params
#         for i in range(len(self.layers)):
#             prev = min(0, i-1)
#             input_shape = (self.layers[prev].n_units,)
#             self.layers[i](input_shape)

#     def add(self, layer):
#         prev = min(0, len(self.layers)-1)

#         # set all layer input params
#         self.layers.append(layer)
#         input_shape = (self.layers[prev].n_units,)
#         self.layers[-1](input_shape)

#     def remove(self, layer_index):
#         del self.layers[layer_index]
#         return self.layers
    
#     def fit(self, X, y, n_epochs, batch_size=64, verbose=True):
#         history_loss = []

#         for epoch in range(n_epochs):
#             batch_error = []
#             for X_batch, y_batch in self._gen_batch(X, y, batch_size):
#                 y_pred = self._forward(X)

#                 # print("called")
#                 # the loss gradient with respect to y_pred
#                 gradient_loss = self.loss_function.gradient(y, y_pred)

#                 # Backpropagate. Update weights
#                 self._backward(gradient_loss)

#                 batch_error.append(
#                     np.mean(self.loss_function(y, y_pred))
#                 )
            
#             history_loss.append(
#                 np.mean(batch_error)
#             )

#             # display trained network state
#             if verbose:
#                 print(f"\r[{epoch}/{n_epochs}] loss:{history_loss[-1]}", end="")

#         # model accuracy
#         if verbose:
#             y_true = np.argmax(y, axis=2)
#             y_pred = np.argmax(self.predict(X), axis=2)
#             accuracy = int(np.mean(np.sum(y_true == y_pred, axis=0)/len(y)) * 100)
#             print(f"\nAccuracy: {accuracy}%\n")

#         return history_loss

#     def predict(self, X):
#         return self._forward(X)

#     def _backward(self, gradient_loss):
#         # update weights in each layer, bind the output to next input gradient loss
#         for layer in reversed(self.layers):
#             gradient_loss = layer.backward(gradient_loss)
            
#     def _forward(self, X):
#         output = X

#         # walk through layers, bind the output to next input
#         for layer in self.layers:
#             output = layer.forward(output)

#         return output
        
#     def _gen_batch(self, X, y=None, batch_size=64):
#         n_samples = X.shape[0]
#         for i in np.arange(0, n_samples, batch_size):
#             begin, end = i, min(i+batch_size, n_samples)
#             if y is not None:
#                 yield X[begin:end], y[begin:end]
#             else:
#                 yield X[begin:end]


# new design

