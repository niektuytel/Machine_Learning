

    # def rnn_cell_backward(self, da_next, cache):
    #     (a_next, a_prev, xt, parameters) = cache
    #     Wax = parameters["Wax"]
    #     Waa = parameters["Waa"]
    #     Wya= parameters["Wya"]
    #     ba= parameters["ba"]
    #     by= parameters["by"]
    #     dtanh = (1 - a_next * a_next) * da_next
    #     dWax = np.dot(dtanh, xt.T)
    #     dxt = np.dot(Wax.T, dtanh)
    #     dWaa = np.dot(dtanh, a_prev.T)
    #     da_prev = np.dot(Waa.T, dtanh)
    #     dba = np.sum(dtanh, keepdims=True, axis=-1)
    #     gradients = {"dxt": dxt, "da_prev": da_prev,"dWax": dWax, "dWaa": dWaa, "dba": dba}
    #     return gradients
        
    # def backward(self, gradient):

    #     # derivatives
    #     self.reset.W_derivative = np.zeros_like(self.reset.W)
    #     self.update.W_derivative = np.zeros_like(self.update.W)
    #     self.hidden.W_derivative = np.zeros_like(self.hidden.W)
    #     self.input.W_derivative = np.zeros_like(self.input.W)
    #     self.output.W_derivative = np.zeros_like(self.output.W)

    #     # backward derivative (chain-rule)
    #     # gradient = loss * self.activation_output.derivative(self.output.values)# not been used 
    #     gradient_next = np.zeros_like(gradient)



    #     # # pass trough neuron
    #     # for t in range(timesteps):
    #     #     self.inputs[:, t] = X[:, t].dot(self.input_W.T) + self.states[:, t-1].dot(self.states_W.T)
    #     #     self.states[:, t] = self.activation(self.inputs[:, t])
    #     #     self.outputs[:, t] = self.states[:, t].dot(self.output_W.T)


    #     # gradient_state = gradient[:, t].dot(self.output_W) * self.activation.derivative(self.inputs[:, t])
    #     # gradient_next[:, t] = gradient_state.dot(self.input_W)
            
    #     #     self.output_Wd += gradient[:, t].T.dot(self.states[:, t])
    #     #     for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
    #     #         self.input_Wd += gradient_state.T.dot(self.layer_input[:, t_])
    #     #         self.states_Wd += gradient_state.T.dot(self.states[:, t_-1])
                
    #     #         gradient_state = gradient_state.dot(self.states_W) * self.activation.derivative(self.inputs[:, t_-1])

    #     # update layer weights/params
    #     _, timesteps, _ = gradient.shape
    #     for t in reversed(range(timesteps)):
    #         gradient_state = gradient[:, t].dot(self.output.W) * self.activation.derivative(self.update.values[:, t])
    #         gradient_next[:, t] = gradient_state.dot(self.update.W)
    #         # gradient_next[:, t] = gradient_state.dot(self.input.W)
            
    #         self.reset.W_derivative += gradient[:, t].T.dot(self.state.values[:, t])
    #         self.update.W_derivative += gradient_state.T.dot(self.layer_input[:, t])
    #         self.hidden.W_derivative += gradient_state.T.dot(self.state.values[:, t-1])

    #         self.output.W_derivative += gradient[:, t].T.dot(self.state.values[:, t])
    #         # self.input.W_derivative += gradient_state.T.dot(self.layer_input[:, t])
    #         self.state.W_derivative += gradient_state.T.dot(self.state.values[:, t-1])

    #         gradient_state = gradient[:, t].dot(self.output.W) * self.activation.derivative(self.update.values[:, t])
    #         # gradient_next[:, t] = gradient_state.dot(self.input.W)

    #         # self.output.W_derivative += gradient[:, t].T.dot(self.state.values[:, t])
    #         # self.input.W_derivative += gradient_state.T.dot(self.layer_input[:, t])
    #         # self.state.W_derivative += gradient_state.T.dot(self.state.values[:, t-1])

    #     self.reset.W  = self.reset.W_optimizer.update(self.reset.W, self.reset.W_derivative)
    #     self.update.W = self.update.W_optimizer.update(self.update.W, self.update.W_derivative)
    #     self.hidden.W = self.hidden.W_optimizer.update(self.hidden.W, self.hidden.W_derivative)
    #     self.state.W = self.state.W_optimizer.update(self.state.W, self.state.W_derivative)
    #     self.output.W = self.output.W_optimizer.update(self.output.W, self.output.W_derivative)

    #     # Return gradient for next layer
    #     return gradient_next
