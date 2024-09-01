'''
This program implements a simple one layer perceptron algorithm.
https://en.wikipedia.org/wiki/Perceptron
Look at the readme for more information.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BatchPerceptron:

    '''
    Input Types for all Functions:

    set_sample_weights -> np.array of shape (self.dim+1,).
                          The first entry is the bias.
    
    
    set_data ->     The training set of the type : 
                    [(x,y)] where x is a np.array with shape (dim,) and y is either 1 or -1.
                    The values 'dim' and 'size' will be corrected.
    '''

    def __init__(self, dim=None, size=None, location=None):
        '''
        THE TYPES OF THE ATTRIBUTES ARE DIFFERENT FROM THE INPUT VALUE-TYPES, 
        BECAUSE THE PROGRAM USES AN EXTRA BIAS INSIDE OF THE VECTORS.
        LOOK AT README FOR CORRECT INPUTS TYPES & USE !!!


        THIS SECTIONS IS ONLY FOR PEOPLE WHO WANT TO UNDERSTAND THE PROGRAM.

        
        self.dim            -> Size of a vector.            (type : Int)
        self.size           -> Size of the training set.     (type : Int)


        self.weights        -> The output weights of the algorithm. It has the shape (dim + 1,), 
                               because it also contains a bias.    (type : np.array)

        self.sample_weights -> sample solution, which can be set by the user
                               with set_sample_weights             (type : np.array)
                               (The dimension of the vector also dim + 1,due to bias.)

        self.data -> The training set of the type : 
                     [(x,y)] where x is a np.array with shape (dim+1,) and y is either 1 or -1.
                     (Different from input type of the data set!)

        The Program calculates the weighted sum : w1 x1 + w2 x2 + .. + b
        using two vectors (b,w1,w2...) and (1,x1,x2...). It will append the bias as 
        the first entry by itself.
        '''
        self.weights = None
        self.sample_weights = None

        self.data = []

        self.dim = dim
        self.size = size

        # If 'location' is defined, the progam will determine the dim and
        # size of the data by itself.
        if location is not None:
            self.__read_data_csv(location)

    def set_sample_weights(self, weights):
        '''
        Defines the sample weights. Takes a np.array of shape (dim+1,)
        '''
        norm = np.linalg.norm(weights)
        self.sample_weights = weights / norm

    def set_data(self, data):
        '''

        '''
        self.data = data

        # Change Values
        self.dim = len(data[0][0])
        self.size = len(data)

        # Insert one at the first entry.
        # Reminder : data is a list of tupels of the format : ([],Int)

        for index, val in enumerate(data):
            self.data[index] = (
                np.insert(val[0], 0, 1),  # new vector
                val[1]                    # Int
            )

    def generate_successfull_data(self):
        '''
        Generates a training set, which is linearly separable.
        The function will create a sample plane if no sample weights were defined.
        '''

        if not self.sample_weights:
            self.set_sample_weights(
                np.random.uniform(-100, 100, self.dim+1)
            )

        counter = 0
        while counter != self.size:

            # Create a random vector of size dim+1 with a one in the first entry.
            # It is used to multiply the bias with it in the dot product.
            random_vector = np.random.uniform(-100, 100, self.dim)
            random_vector = np.insert(random_vector, 0, 1)

            dot = np.dot(random_vector, self.sample_weights)

            # Cross product greater than 0 implies that the vector is not on the plane.
            if dot > 0:
                self.data.append(
                    (random_vector, 1)
                )
            elif dot < 0:
                self.data.append(
                    (random_vector, -1)
                )
            else:
                continue
            counter += 1

        return self.data

    def run(self):
        '''
        Runs the Algorithm.
        '''

        # Base case
        weights = np.zeros(self.dim+1)

        all = False
        while not all:

            all = True
            for val in self.data:

                # We change the weights until this condition is false for all elements of the training set.
                if np.dot(val[0], weights) * val[1] <= 0:
                    weights += val[1] * val[0]
                    all = False
                    break

        norm = np.linalg.norm(weights)
        self.weights = weights / norm
        return self.weights

    def __read_data_csv(self, location):
        '''
        Read the data out of a csv file.
        The program will append the bias automatically.
        '''

        # Read the CSV file
        df = pd.read_csv(location)

        # Separate x (vectors) and y (numbers)
        x = df.iloc[:, :-1].values.tolist()  # Convert x to a list of lists
        y = df.iloc[:, -1].values.tolist()   # Convert y to a list

        if x == []:
            raise TypeError('Dimension of the X vector has to be atleast 1.')

        self.dim = len(x[0])
        self.size = len(y)

        # Append bias at first pos of vectors.
        for index in range(0, len(x)):
            x[index] = np.insert(x[index], 0, 1)

        self.data = list(zip(
            np.array(x),
            np.array(y)
        ))
        print(self.data)

    def error_to_sample(self):
        '''
        Calculates a Error if a sample exists.
        '''

        if self.sample_weights is None:
            raise TypeError("No sample weights were defined.")

        return sum(
            np.abs(self.sample_weights - self.weights)
        )

    def visualise(self):
        '''

        Will plot a 3D Graph with the calculated plane and the vectors of the trainingsset.
        It will plot at max 200 samples.
        The dimension of the vectors has to be 3!

        '''
        if self.dim != 3:
            raise ValueError("Dimension needs to be 3.")
        if self.weights is None:
            raise TypeError("Run the Algorithm first, to create a plane!")

        # Split the data to plot it & calculate max values for each direction
        blue = []
        red = []

        max_x = float('-inf')
        max_y = float('-inf')
        max_z = float('-inf')

        for i, val in enumerate(self.data):
            # Use only the first 200 examples.
            if i > 200:
                break

            if val[1] == 1:
                blue.append(val[0].tolist())
            else:
                red.append(val[0].tolist())

            # calculate max values. val[0][0] is the bias
            if val[0][1] > max_x:
                max_x = val[0][1]
            if val[0][2] > max_y:
                max_y = val[0][2]
            if val[0][3] > max_z:
                max_z = val[0][3]

        red = np.array(red)
        blue = np.array(blue)

        # Calculate values for the plane
        b = self.weights[0]
        A = self.weights[1]
        B = self.weights[2]
        C = self.weights[3]

        # Calculate the plane.
        # Exception when C = 0, due to the division by C.
        if C != 0:
            x = np.linspace(-max_x, max_x, 100)
            y = np.linspace(-max_y, max_y, 100)
            x, y = np.meshgrid(x, y)
            z = (-A * x - B * y - b) / C

        else:
            x = np.linspace(-max_x, max_x, 100)
            z = np.linspace(-max_z, max_z, 100)
            x, z = np.meshgrid(x, z)
            y = (-A * x - b) / B

        # Plot everything
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100)

        ax.scatter(red[:, 1], red[:, 2], red[:, 3],
                   color='red', label='Set 1', s=50)
        ax.scatter(blue[:, 1], blue[:, 2], blue[:, 3],
                   color='blue', label='Set 1', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


# Test = BatchPerceptron(location='C:/Users/Luca/Desktop/data.csv')
# Test.run()
# Test.visualise()

'''
Test = BatchPerceptron(dim = 2, size = 3)
data = [
    ([1,2,1],1),
    ([4,-1,1],-1),
    ([0.5, 1, -0.2], 1)
]
Test.set_data(data)
Test.run()
Test.visualise()
'''

Test = BatchPerceptron(dim=3, size=2000)
Test.generate_successfull_data()
Test.run()
print(Test.error_to_sample())
Test.visualise()
