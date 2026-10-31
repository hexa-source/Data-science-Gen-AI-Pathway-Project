# import numpy library
import numpy as np
import time

# create class for multille neuron with two weghts & two biases 

class MLP_ReSoft:   # this module combines both ReLU activation function and Softmax for multiclass classification

    # initiate the wieghts, bias, learning rate, hidden_state, n_classes & input_size

    def __init__(self, n_inputs, n_hidden, n_classes, learning_rate = 0.01, epochs = 100):

        rng = np.random.default_rng()
        self.w_1 = rng.standard_normal((n_inputs, n_hidden))    
        self.b_1 = np.zeros((1,n_hidden))

        self.w_2 = rng.standard_normal((n_hidden, n_classes))
        self.b_2 = np.zeros((1,n_classes))

        self.lr = learning_rate
        self.ep = epochs

    # Define the activation function using Rectified Linear Unit function & softmax for multiclass classification

    def ReLU(self, z):
        return np.maximum(0,z)

    def ReLU_derivative(self, z):
        return (z>0).astype(float)

    def Softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
        return exp_z/np.sum(exp_z, axis = 1, keepdims = True)


    # Define the forward propagation function


    def forward_pass(self, X):

        self.z_1 = np.dot(X, self.w_1) + self.b_1
        self.a_1 = self.ReLU(self.z_1)

        self.z_2 = np.dot(self.a_1, self.w_2) + self.b_2
        self.output = self.Softmax(self.z_2)

        return self.output

    def CategoryCrossEntropy(self, y, output):
            np.clip(output, 1e-12, 1.0)
            loss = -np.mean(np.log(output[np.arange(len(y)), y]))
            return loss

    # Define the backpropagation function

    def backprop(self, X, y):
        # for n_inputs
        n = X.shape[0]

        d_output = np.eye(y.nunique())[np.asarray(y).ravel()]
        dz_2 = (self.output - d_output)/n
        dw_2 = self.a_1.T @ dz_2
        db_2 = np.sum(dz_2, axis = 0, keepdims = True)/n

        da_1 = dz_2 @ self.w_2.T
        dz_1 = da_1 * np.where(self.a_1>0, 1, 0)
        dw_1 = X.T @ dz_1
        db_1 = np.sum(dz_1, axis = 0, keepdims = True)/n

        # update the weights and biases
        self.w_2 -= self.lr * dw_2
        self.b_2 -= self.lr * db_2
        self.w_1 -= self.lr * dw_1
        self.b_1 -= self.lr * db_1


    # define the fit function
    def fit(self, X, y):
        for epoch in range(self.ep):
            
            forward = "->" * 60
            print(f"Feeding data forward::::{forward}")
            # time.sleep(3)
            prediction = self.forward_pass(X)
            
            print("-------------------------Calculating CrossEntropy loss-------------------------------")
            # time.sleep(1)
            loss = self.CategoryCrossEntropy(y, prediction)
            backward = "<-" * 60
            

            print(f"{backward}::::propagating backward to update weights and biases")
            # time.sleep(5)
            self.backprop(X, y)
            print("=" * 60)
            # time.sleep(10)
            print(f"Epoch: {epoch + 1}|",
                  f"Loss: {loss:.4f}|"
                  )

        return self

    # define the predict function

    def predict_proba(self, X):
        return self.forward_pass(X)


    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)


    def accuracy_score(self, y, y_pred):
        return np.mean(y_pred == y)




if __name__ == "__main__":
    
    X = np.array([[0,0,1,1],
                  [0,1,0,1]])

    Y = np.array([[0,1,1,0]]) 
    model = MLP_ReSoft(n_inputs = 21, n_hidden = 8,learning_rate = 0.5, epochs = 10000)
    model.fit(X, Y)
    pred = model.predict(X)
    print(pred)


    print("""::::::::::::::::::::::::::::::::::::::::::
    \n.........................................\nReLU_Softmax\nmodel run successfully\n""")



