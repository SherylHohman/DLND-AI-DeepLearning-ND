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
        
        '''
        print ('nodes   - input, hidden, output ',                      \
               self.input_nodes, self.hidden_nodes, self.output_nodes   \
              )  # 3 2 1
        print ('weights - input_to_hidden_shape, hidden_to_output_shape: ',            \
               self.weights_input_to_hidden.shape, self.weights_hidden_to_output.shape \
              )  # (3, 2) (2, 1)
        '''
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x : 1/(1 + np.exp(-x))  
        

    def train(self, features, targets):
        #print('features.shape: ', features.shape)
        #print('targets.shape : ', targets.shape)
        
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
        ### Forward pass ###
        
        # signals into Hidden layer
        hidden_inputs  = np.dot(X, self.weights_input_to_hidden)
        
        # signals from Hidden layer
        # activation: sigmoid aka self.activation_function; 
        hidden_outputs = self.activation_function(hidden_inputs) 

        # signals into final Output layer
        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        
        # signals from final Output layer
        # activation function: f(x) = x
        final_outputs = final_inputs  
        
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
        ### Backward pass ###

        ## TODO: Output error
        # Output layer error is the difference between desired target and actual output.
        
        error = y - final_outputs
        
        #print("error shape:", error.shape)                          # (1,)
                
        ### TODO: Backpropagated error terms ###
        
        ## TODO: Calculate the output layer's Backpropogated error term ##
        #  NOTICE: ACTIVATION FOR OUTPUT IS NOT SIGMOID, IT IS F(X)=X       
        # output activation: f(x) = x
        # output activation_prime: 1 
        # output_error_term: error * activation_prime === error * 1 === error

        output_error_term = error                                   # (1,)
        
        #print("output_error_term shape:", output_error_term.shape)
        
        ## TODO: Calculate the hidden layer's contribution to the error ##

        hidden_error = np.dot(output_error_term[:, None],           #(1,1)
                              self.weights_hidden_to_output.T       #(1,2)
                             )
        
        #print("hidden_error shape:", hidden_error.shape)            # (1, 2)
        
        
        ## TODO: Calculate the hidden layer's Backpropogated error term ##        

        # gradient of the activation function, sigmoid is: 
        # sigmoid_prime is: layer_output * (1-layer_output)
        # layer error   is: error * sigmoid_prime(layer_input)

        hidden_error_term = hidden_error * (hidden_outputs) * (1-hidden_outputs)
        
        #print("hidden_error_term shape:", hidden_error_term.shape)  # (1, 2)
                
        ### delta_weights_layer: learningRate * layer_error_weighted * layer_input_values ###
        
        ## Weight step (hidden to output) ##       
        delta_weights_h_o += self.lr * hidden_outputs[:, None] \
                                     * output_error_term             #[None, :]   # ho(2,)er(1,)
        
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
        '''
        #print("_delta_weights_h_o shape:", delta_weights_h_o.shape)  # (2, 1)
        
        ## Weight step (input to hidden) ##        
        delta_weights_i_h += self.lr * X[:, None]  \
                                     * hidden_error_term
        
        #print("delta_weights_i_h shape:", delta_weights_i_h.shape)
        
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
               
        
        # HIDDEN LAYER
        # signals into Hidden Layer
        hidden_inputs  = np.dot(features, self.weights_input_to_hidden)        
        
        # signals from Hidden Layer
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        ## OUTPUT LAYER        
        # signals into final Output Layer
        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output)         
        
        # signals from final Output Layer
        final_outputs = final_inputs#self.activation_function(final_inputs)  
        
        return final_outputs
    

#########################################################
# Set your hyperparameters here
##########################################################
'''
train_features shape: (15435, 56)
  val_features shape: ( 1440, 56)
 test_features shape: (  504, 56)
'''

iterations = 1200     # batch size: 128, 120 iterations / pass of the entire dataset (on average)
learning_rate = 0.015 # 56 ** -.5 = .134
hidden_nodes = 6      # features: 56, hidden nodes: (n_features-n_outputs)/2, try: 28
output_nodes = 1      # I want an output of a single number: total number of rentals at each (hour)
print('params: ',iterations, learning_rate, hidden_nodes, output_nodes)




'''
# TEST RESULTS, TUNING THE PARAMETERS:
# 100, 0.1,   2, 1  :Progress: 99.0% ... Training loss: 0.952 ... Validation loss: 1.365
# 100, 0.01,  2, 1 *:Progress: 99.0% ... Training loss: 0.382 ... Validation loss: 0.468
# 100, 0.001, 2, 1  :Progress: 99.0% ... Training loss: 0.732 ... Validation loss: 1.278

# 200, 0.01,  2, 1  :Progress: 99.5% ... Training loss: 0.331 ... Validation loss: 0.479
# 200, 0.01,  3, 1  :Progress: 99.5% ... Training loss: 0.277 ... Validation loss: 0.453
# 100, 0.01,  3, 1  :Progress: 99.0% ... Training loss: 0.315 ... Validation loss: 0.500
# 150, 0.01,  2, 1  :Progress: 99.3% ... Training loss: 0.278 ... Validation loss: 0.466
# 200, 0.05,  2, 1  :Progress: 99.5% ... Training loss: 0.272 ... Validation loss: 0.434

# 100, 0.05, 27, 1  :Progress: 99.0% ... Training loss: 0.951 ... Validation loss: 1.367
# 500, 0.1,  27, 7  :Progress: 99.8% ... Training loss: 1.142 ... Validation loss: 1.367
# 240, 0.1,  26, 1  :NAN
# 100, 0.01, 26, 1  :Progress: 99.0% ... Training loss: 0.998 ... Validation loss: 1.367
# 100, 0.1,  26, 1  :Nan
# 100, 0.8,  26, 1  :Nan
# 100, 0.1,   6, 1  :Progress: 99.0% ... Training loss: 6.969 ... Validation loss: 7.359 
                     Terrible Step-Function-like output, 
# 100, 0.05, 20, 1  :Progress: 99.0% ... Training loss: 0.974 ... Validation loss: 1.367 
                     Terrible L-like output. Flat Line
# 100, 0.1,  20, 1  :Progress: 99.0% ... Training loss: 8.073 ... Validation loss: 7.580 
                     Terrible Backwards-L, and wide heartrate, way below the line
# 100, 0.01,  6, 1  :Progress: 99.0% ... Training loss: 0.326 ... Validation loss: 0.555
                     a graph offset
# 500, 0.01,  6, 1  :Progress: 99.8% ... Training loss: 0.178 ... Validation loss: 0.327 
                     a graph offset
# 500, 0.05,  6, 1  :Progress: 99.8% ... Training loss: 0.973 ... Validation loss: 1.367 
                     Terrible L-like output. Flat Line
# 800, 0.02,  6, 1  :Progress: 99.9% ... Training loss: 0.109 ... Validation loss: 0.192
 2400, 0.02,  6, 1  :Progress: 100.0% ... Training loss: 0.068 ... Validation loss: 0.138
 4800, 0.02?, 6?,1  :Progress: 100.0% ... Training loss: 0.064 ... Validation loss: 0.145
                     no real diff from prev.. numbers hovered without changing thru numerous epochs
 4800, 0.01, 12, 1 *:Progress: 100.0% ... Training loss: 0.057 ... Validation loss: 0.134
                     slight improvements. hand't settled yet. still slowly improving
 5600, 0.01, 24, 1  :Progress: 100.0% ... Training loss: 0.975 ... Validation loss: 1.367
                     Terrible L-like output. Flat Line. Too small LR or too many nodes
 3600, 0.04, 24, 1  :Progress: 100.0% ... Training loss: 0.901 ... Validation loss: 1.353                     
 7200, 0.01, 24, 1  :Progress: 100.0% ... Training loss: 0.886 ... Validation loss: 1.315
                     Terrrible, L, but interesting prediction graph
 7200, 0.01, 12, 1  :Progress: 100.0% ... Training loss: 0.066 ... Validation loss: 0.161
                     ?? not as good as 4800, .01, 12, 1 ???
 9600, 0.02, 12, 1 *:Progress: 100.0% ... Training loss: 0.511 ... Validation loss: 0.618
 
                     Weird. It tracks well - probably OVERFITTING 
                     --EXCEPT that its offset -by like 200 maybe? 
                     Subratact about 200 from the "Lows/Minimums" for prediction to match actual.
                     Highs are closer (except +/- xmas), subtract maybe 50? so P Highs == actual max.
                     22Dec - 31Dec likely won't ever match.
 9960, 0.18, 14, 1   NaN.  Seems 12 Nodes is the Max.  
                     My attempt to Return 1 from sigmoid, if Nan, didn't work.
                     I could try to figure it out. But 
                     1) more nodes is probably not the right solution (overcomplicated)
                     2) Time is better spent wrapping this up, and continuing onto the Next Lesson.
                     3) it's already been hinted at other ways of keeping weights in a useful range.
                     4) Leaky_abstraction
                     5) Getting into overfitting region anyway.
                     
 9960, 0.018, 10, 1 *:Progress: 100.0% ... Training loss: 0.069 ... Validation loss: 0.132
 
 4800, 0.018, 10, 1 *:Progress:   0.1% ... Training loss: 1.970 ... Validation loss: 1.707 
                      OOPS-WRONG STATT
                      
  500, 0.01,   6, 1 *:Progress:  99.8% ... Training loss: 0.223 ... Validation loss: 0.332
 7200, 0.015,  8, 1 *:Progress: 100.0% ... Training loss: 0.060 ... Validation loss: 0.148 
 9600, 0.015,  8, 1  :Progress: 100.0% ... Training loss: 0.058 ... Validation loss: 0.184
                      This model began increasing Validation loss about 40-45% through the training
                      Meanwhile, Training Loss held relatively steady.  Shall run again, but stop at around
                      9600*(.40) to 96*(.45) = 3840 to 4320. Let's try 4200.. 
 4200, 0.015,  8, 1  :Progress: 100.0% ... Training loss: 0.079 ... Validation loss: 0.204 
 4200, 0.015,  6, 1**:Progress: 100.0% ... Training loss: 0.061 ... Validation loss: 0.139 
 4200, 0.010,  6, 1 *:Progress: 100.0% ... Training loss: 0.062 ... Validation loss: 0.154
 4200, 0.015, 12, 1  :Progress: 100.0% ... Training loss: 0.207 ... Validation loss: 0.379 
 4200, 0.015,  6, 1**:Progress: 100.0% ... Training loss: 0.059 ... Validation loss: 0.146

-----
 4200, 0.015,  6, 1  :Progress: 100.0% ... Training loss: 0.059 ... Validation loss: 0.141
 4200, 0.3,   10, 1  : NaN
 4200, 0.1,    8, 1  :Progress: 100.0% ... Training loss: 0.936 ... Validation loss: 1.355
 4200, 0.05,   8, 1  :Progress:  92.8% ... Training loss: 0.974 ... Validation loss: 1.367
                      Useless - mostly didn't change losses between iterations
 4200, 0.015,  6, 1  :Progress: 100.0% ... Training loss: 0.067 ... Validation loss: 0.163
                      Looks good, but V-Loss wasn't decreasing, so perhaps quit sooner
 3600, 0.015,  6, 1  :Progress: 100.0% ... Training loss: 0.062 ... Validation loss: 0.148
 2400, 0.02,   6, 1  :Progress: 100.0% ... Training loss: 0.098 ... Validation loss: 0.215
                      NOPE - doesn't pass. too few epochs, or lr not ideal
 3600, 0.02,   6, 1  :Progress: 100.0% ... Training loss: 0.115 ... Validation loss: 0.222
                      NOPE - VL jumped up right at the end
 2400, 0.015,  6, 1 *:Progress: 100.0% ... Training loss: 0.074 ... Validation loss: 0.150
 1000, 0.015,  6, 1  :Progress: 99.9% ... Training loss: 0.075 ... Validation loss: 0.160
 1200, 0.015,  6, 1  :Progress: 99.9% ... Training loss: 0.069 ... Validation loss: 0.161
                                            
'''

# Required: The training loss is below 0.09 and the validation loss is below 0.18.