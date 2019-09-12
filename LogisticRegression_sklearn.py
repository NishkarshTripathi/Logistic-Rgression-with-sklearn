from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import metrics

# Load the dataset
digits = load_digits()

#print(digits.data.shape)

# Visualizing image data
#plt.figure(figsize=(20, 4))
#for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
#    plt.subplot(1, 5, index+1)
#    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
#    plt.show()

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticreg = LogisticRegression()
print("Training Model....")
logisticreg.fit(x_train, y_train)
print("Model Trained.")

# Measure the performance of  the model
score = logisticreg.score(x_test, y_test)
print("Score: ", score)

# Visualizing performance through Confusion Matrix
predictions = logisticreg.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(9, 9))
sb.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title = 'Accuracy Score {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()