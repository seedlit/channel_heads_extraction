import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix

# loading the training data
traindata = np.loadtxt("PATH_TO_TRAINING_DATA", delimiter = ",")

# storing the features into the variable 'X'. Here some features are - upstream area (A), slope(S), Plan Curvature, Profile Curvature, etc.
X = traindata[:,2:8]

# performing feature scaling if required. Recommended if training on a neural network
"""for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - min(X[:,i]))/(max(X[:,i]) - min(X[:,i]))"""

# storing the ground truth values (1 = channel head, 0 = not a channel head) into the variable 'Y'
Y = traindata[:,8]

# The datatset is highly imbalanced. Hence oversampling the data. (Undersampling leads to poorer performance)
resample = RandomOverSampler()
X_resampled, Y_resampled = resample.fit_sample(X, Y)

# Fitting a decision tree on our training dataset
decision_tree = RandomForestRegressor(n_estimators=8, max_depth=6, max_features=5, random_state=42)
#decision_tree = DecisionTreeRegressor(max_depth=6)
decision_tree.fit(X_resampled, Y_resampled)
print(decision_tree.score(X_resampled, Y_resampled))

# Hyperparameters will be tuned using validation set
validationdata = np.loadtxt("PATH_TO_VALIDATION_DATA", delimiter = ",")
validationX = validationdata[:,2:8]

# performing feature scaling if required
"""for i in range(validationX.shape[1]):
    validationX[:,i] = (validationX[:,i] - min(validationX[:,i]))/(max(validationX[:,i]) - min(validationX[:,i]))"""

# predicting on the validation set
validationPredictions = np.round(decision_tree.predict(validationX))
# validation groundtruth
validationY = validationdata[:,8]
print(decision_tree.score(validationX, validationY))
print(confusion_matrix(validationY, validationPredictions))
print(classification_report(validationY, validationPredictions))

# predicting on test dataset
testdata = np.loadtxt("PATH_TO_TEST_DATA", delimiter = ",")
testX = testdata[:,2:8]

# performing feature scaling if required
"""for i in range(testX.shape[1]):
    testX[:,0] = (testX[:,i] - min(testX[:,i]))/(max(testX[:,i]) - min(testX[:,i]))"""

testPredictions = np.round(decision_tree.predict(testX))
testY = testdata[:,8]
print(decision_tree.score(testX, testY))
print(confusion_matrix(testY, testPredictions))
print(classification_report(testY, testPredictions))
 
# exporting the data
PointId = testdata[:,0]
ChannelHeadActual = testdata[:,8]
channelHeadDT = pd.DataFrame({ 'PointId': PointId,
                            'ChannelHeadPredicted': testPredictions, 'ChannelHeadActual' : ChannelHeadActual })
channelHeadDT.to_csv("OUTPUT_DIR", index=False)
