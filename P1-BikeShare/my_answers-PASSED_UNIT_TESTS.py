import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes  =  input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        print ('nodes   - input, hidden, output ',                      \
               self.input_nodes, self.hidden_nodes, self.output_nodes   \
              )  # 3 2 1
        print ('weights - input_to_hidden_shape, hidden_to_output_shape: ',            \
               self.weights_input_to_hidden.shape, self.weights_hidden_to_output.shape \
              )  # (3, 2) (2, 1)
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        self.activation_function = lambda x : 1/(1 + np.exp(-x))  
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on a batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            final_outputs,      \
            hidden_outputs =    \
                                self.forward_pass_train(X)  
            
            delta_weights_i_h,  \
            delta_weights_h_o = \
                                self.backpropagation(final_outputs, hidden_outputs, 
                                                     X, y, 
                                                     delta_weights_i_h, 
                                                     delta_weights_h_o
                                                     )
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        
        ## TODO: Hidden layer - Replace these values with your calculations.
        # signals into hidden layer
        hidden_inputs  = np.dot(X, self.weights_input_to_hidden)
        print("hidden_inputs shape:", hidden_inputs.shape)         # (2,)
        
        # signals from hidden layer
        # activation: sigmoid, activation_prime: output * (1-output)
        hidden_outputs = self.activation_function(hidden_inputs) 
        print("hidden_outputs shape:", hidden_outputs.shape)       # (2,)

        ## TODO: Output layer - Replace these values with your calculations.
        # signals into final output layer
        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        print("final_inputs shape:", final_inputs.shape)           # (1,)
        
        # signals from final output layer
        # activation: f(x), activation_prime: 1
        final_outputs = final_inputs  
            #X-NOPE: #final_outputs = self.activation_function(final_inputs) 
        print("final_outputs shape:", final_outputs.shape)         # (1,)
        
        return final_outputs, hidden_outputs
    

    def backpropagation(self, final_outputs, hidden_outputs, 
                        X, y, 
                        delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        ## TODO: Output error
        # Output layer error is the difference between desired target and actual output.
        # activation, output layer: x + y  => activation_prime: dy == y - y_hat
        error = y - final_outputs
        print("error shape:", error.shape)                          # (1,)
                
        ## TODO: Backpropagated error terms - Replace these values with your calculations.
        
        # gradient of the activation function, sigmoid: layer_output * (1-layer_output)
        # and the layer error is: error * sigmoid_prime(layer_input)

        # TODO: Calculate the output layer's Backpropogated error term
        # NOTICE: ACTIVATION FOR OUTPUT IS NOT SIGMOID, IT IS F(X)=X
        # output_error_term = error * f_prime(x)
        # output activation: f(x) = x
        # output activation_prime: 1 
        # output_error_term: error * activation_prime === error * 1 === error
        output_error_term = error                                   # (1,)
        # NOPE: WRONG ACTIVATION FOR THIS PROBLEM: #output_error_term = error * y * (1-y) #(1,)
        print("output_error_term shape:", output_error_term.shape)
        
        # TODO: Calculate the hidden layer's contribution to the error
        #hidden_error = np.dot(output_error_term,            \
        #                      self.weights_hidden_to_output \
        #                     )
        # output_error_term is a scaler, so not using the dot product
        #hidden_error = output_error_term * self.weights_hidden_to_output 
        hidden_error = np.dot(output_error_term[:, None],           #(1,1)
                              self.weights_hidden_to_output.T       #(1,2)
                             )
        print("hidden_error shape:", hidden_error.shape)            # (1, 2)
        
        # TODO: Calculate the hidden layer's Backpropogated error term
        hidden_error_term = hidden_error * (hidden_outputs) * (1-hidden_outputs)
        print("hidden_error_term shape:", hidden_error_term.shape)  # (1, 2)
                
        ## delta_weights_layer: learningRate * layer_error * layer_input_values
        # Weight step (hidden to output)
        
        delta_weights_h_o += self.lr * hidden_outputs[:, None] \
                                  * output_error_term#[None, :]   # ho(2,)er(1,)
        
        '''
        # These 3 versions SAME
        delta_weights_h_o_1 = self.lr * hidden_outputs[:, None] \
                                   * output_error_term            # ho(2,)er(1,)        
        delta_weights_h_o_2 = self.lr * hidden_outputs[:, None] \
                                   * output_error_term[:, None]   # ho(2,)er(1,)        
        delta_weights_h_o_3 = self.lr * hidden_outputs[:, None] \
                                   * output_error_term[None, :]   # ho(2,)er(1,)
        #This 1 version BAD
        delta_weights_h_o_4 = self.lr * hidden_outputs[None, :] \
                                   * output_error_term#[None, :]   # ho(??)er(??)->(1, 2)
        print("_delta_weights_h_o_1:", delta_weights_h_o_1)  # (2, 1)
        print("_delta_weights_h_o_2:", delta_weights_h_o_2)  # (2, 1)
        print("_delta_weights_h_o_3:", delta_weights_h_o_3)  # (2, 1)
        print("_delta_weights_h_o_4:", delta_weights_h_o_4, delta_weights_h_o_4.shape)  # (1, 2)
        '''
        print("_delta_weights_h_o shape:", delta_weights_h_o.shape)  # (2, 1)
        
        # Weight step (input to hidden)
        delta_weights_i_h += self.lr * X[:, None] * hidden_error_term
        #delta_weights_i_h += self.lr * hidden_error_term * X[:, None] #X:(3,) er:(1, 2)
        print("delta_weights_i_h shape:", delta_weights_i_h.shape)
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += delta_weights_h_o 
        
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h 

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        
        ## TODO: Hidden layer - Replace these values with the appropriate calculations.
        # signals into hidden layer
        hidden_inputs  = np.dot(features, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        ## TODO: Output layer - Replace these values with the appropriate calculations.        
        # signals into final output layer
        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        
        # signals from final output layer
        final_outputs = final_inputs#self.activation_function(final_inputs)  
        
        return final_outputs
    

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
