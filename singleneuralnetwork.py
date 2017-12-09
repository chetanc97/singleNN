from  numpy import exp ,  array ,  random ,  dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.weights = 2 *  random.random((3 , 1)) - 1


    def __sigmoid(self, x):
        return  1/(1 +  exp(-x))
    def  __sigmoid_derivative(self,x):
        return x * (1-x)
    def train(self ,training_set_inputs ,training_set_outputs ,number_of_iterations ):
        for iteration in range(number_of_iterations):
            output = self.predict (training_set_inputs)

            error = training_set_outputs -  output

            adjustment =  dot(training_set_inputs.T , error * self.__sigmoid_derivative(output))

            self.weights +=adjustment

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))



if __name__ ==  "__main__":
#initialize  a single neuron
    neural_netowrk  =NeuralNetwork()

print ('Lets start with Some Random  weights ')
print (neural_netowrk.weights)


training_set_inputs  =   array([[0,0,1] , [1,1,1] , [1,0,1] , [0,1,1] ])
training_set_outputs =   array([[0,1,1,0]]).T

# train network
neural_netowrk.train(training_set_inputs ,  training_set_outputs , 10000)

print ('New  weights  after training  ;-)')
print (neural_netowrk.weights)

#Test the NN

print ('The Neural network  predicts ')
print (neural_netowrk.predict(array([1,1,0])))
