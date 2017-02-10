import numpy

# Start by creating a design matrix for regression with m = 150 examples, each of dimension d = 75.
# We will choose a true weight vector theta that has only a few non-zero components:

# 1. Construct a random design matrix X 
X = numpy.random.rand(150,75)

# 2. Construct a true weight vector theta. Set the first 10 component of theta to 10 or -10 arbitrarily and all the other components to zero.
t = [-1,1]
theta_1 = numpy.random.choice(t,10)*10
theta_2 = numpy.zeros(65)
theta = numpy.concatenate((theta_1,theta_2), axis=0)

# 3. Construct a vector y = X theta + epsilon, where epsilon is a random noise vector
epsilon = 0.1*numpy.random.randn(150)
y = numpy.dot(X,theta)+epsilon

# 4. Split the dataset by taking the first 80 points for training, the next 20 points for validation,
# and the last 50 points for testing
X_training = X[0:80,:]
y_training = y[0:80]
X_validation = X[80:100,:]
y_validation = y[80:100]
X_testing = X[100:150,:]
y_testing = y[100:150]

# 5. Save six files to local
numpy.savetxt('X_train.txt', X_training) 
numpy.savetxt('y_train.txt', y_training) 
numpy.savetxt('X_valid.txt', X_validation)
numpy.savetxt('y_valid.txt', y_validation) 
numpy.savetxt('X_test.txt', X_testing) 
numpy.savetxt('y_test.txt', y_testing) 