#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
import numpy as np
import sklearn.neural_network
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([0,1,1,0])

model = sklearn.neural_network.MLPClassifier(
                activation='logistic',
                max_iter=100,
                hidden_layer_sizes=(2,),
                solver='lbfgs')
model.fit(inputs, expected_output)
print('predictions:', model.predict(inputs))
print(model.score(inputs,expected_output ))
