class LinearRegression():
    """Basic implementation of linear regression model with gradient descent
    
    The main goal of this implementation it's explicit explanation how
    it works. For best understanding we even won't be using numpy library.

    Args:
        iterations (int): how many gradient descent steps wish to do
        learning_rate (float): decrease learning step to make learning procces
            more smoose.
    """
    
    def __init__(self, iterations=20, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    def fit(self, features, target):
        '''Model training

        We calculate how many features entered. Initializing zero weights for
        each feature and initializing zero bias. Then we run through gradient 
        descent several times.

        Args:
            features (list): matrix with training data, where in columns we have
                different features (item properties), and in rows we have 
                different items (observations).
            target (list): what we desire to predict
        '''
        self.features = features
        self.target = target
        self.features_size = len(features[0])
        self.len_dataset = len(target)
        self.weights = [0] * self.features_size 
        self.bias = 0
        for _ in range(self.iterations):
            self.gradient_descent()
    
    def predict(self, features):
        '''Make prediction by model

        Args:
            features (list): matrix with training data, where in columns we have
                different features (item properties), and in rows we have 
                different items (observations).
        Returns:
            predicted_targets (list): predicted targets by the model
        '''
        predicted_targets = []
        for row in range(len(features)):
            predicted_target = 0
            for col in range(self.features_size):
                predicted_target += features[row][col] * self.weights[col]
            predicted_target += self.bias
            predicted_targets.append(predicted_target)
        return predicted_targets
    
    def gradient_descent(self):
        '''Calculating gradient vector and apply it to weights and bias.

        Gradient it's vector of derivatives from cost function. Gradient 
        descent it's applying this vector to weights and bias. For best 
        understanding this process I strongly recommended you wath this video:
        https://www.youtube.com/watch?v=sDv4f4s2SB8 
        '''
        predicted = self.predict(self.features)
        mse_d = self.mean_squared_error_derivative(self.target, predicted)

        gradient = []
        for col in range(self.features_size):
            delta_for_weight = 0
            for row in range(self.len_dataset):
                delta_for_weight += (
                    self.features[row][col] * mse_d * self.learning_rate
                )
            delta_for_weight = delta_for_weight / self.len_dataset   
            gradient.append(delta_for_weight)
        
        weights_updated = []
        for i in range(self.features_size):
            weight_updated = self.weights[i] - gradient[i] * self.learning_rate
            weights_updated.append(weight_updated)
        self.weights = weights_updated
        self.bias = self.bias - mse_d * self.learning_rate
        rmse = self.root_mean_squared_error(
            self.target, self.predict(self.features)
        )
        print(f'RMSE is {rmse:.2f}')
    
    def mean_squared_error_derivative(self, y_true, y_predicted):
        '''Calculeting derivative from MSE

        Mean squred error it's common used cost function for learning models.
        The good news for us that this function has derivatives whic we use on
        our lerning.
        Read more about this function by link below:
        https://en.wikipedia.org/wiki/Mean_squared_error

        Args:
            y_true (list): real target values
            y_predicted (list): predicted target values by model

        Returns:
            mse_derivative (float): derivative from mean squred error, which 
                show us increasin error or decrising. We can us this derivative
                to update your weights and bias.
        '''
        squared_error_derivatives = []
        for i in range(self.len_dataset):
            squared_error_derivative = -(2 * (y_true[i] - y_predicted[i]))
            squared_error_derivatives.append(squared_error_derivative)
        # mse_derivative (mean squared error derivative)
        mse_derivative = sum(squared_error_derivatives) / self.len_dataset
        return mse_derivative

    def root_mean_squared_error(self, y_true, y_predicted):
        '''Root squared from mean squared error.

        This function was described above. But here we calculate the root from
        MSE to understand mean error in real quantity not squared.
        Args:
            y_true (list): real target values
            y_predicted (list): predicted target values by model

        Returns:
            root_mse (float): root squared from MSE
        '''
        squred_error = 0
        for i in range(len(y_true)):
            # explanation: 3 ** 2 == 3^2 == 3 * 3 == 9
            squred_error += (y_true[i] - y_predicted[i]) ** 2
        mean_squred_error = squred_error / len(y_true)
        # mse - mean squred error
        root_mse = mean_squred_error ** 0.5
        return root_mse

# Test model    
 
if __name__ == '__main__':
    # Dataset with information about a flats in Moscow. Features it's 
    #information about square in meters and rooms. Target it's price in USD.
    features_train = [
        [33, 1],
        [61, 3],
        [40, 1],
        [50, 2],
        [28, 1],
        [80, 3],
        [60, 2]
    ]
    target_train = [
        65000,
        115000,
        100000,
        100000,
        56000,
        200000,
        120000
    ]
    model = LinearRegression()
    model.fit(features_train, target_train)

    features_test = [
        [35, 1],
        [65, 3]
    ]
    predicted = model.predict(features_test)
    print()
    print(f'Test data:')
    print(features_test)
    print()
    print('Predicted values')
    print(predicted)
