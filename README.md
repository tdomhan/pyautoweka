pyautoweka
==========

Description
-----------

pyautoweka is a python wrapper for [Auto-WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/), a Java application for algorithm selection and hyperparameter optimizations, that is build on [WEKA](http://www.cs.waikato.ac.nz/ml/weka/). 


Installation
------------

Download, go to the project sources and install:
```
git clone git@github.com:tdomhan/pyautoweka.git
cd pyautoweka
python setup.py install
```

Running an experiment
--------------------

AutoWeka for python.

```python
import pyautoweka

#Create an experiment
experiment = pyautoweka.Experiment(tuner_timeout=9000)
```
`tuner_timeout` is the time the optimization will run in seconds. So e.g. 9000 seconds = 2.5 hours. The longer you run the optimization, the better of course. (Note that the `experiment` object has an interface similar to sklearn classifiers.) 

First we need to load some data. Let's for example the the famous [Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris). Download it using [this link](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

Let's load it into python:

```python
#load the data:
import numpy as np
import random

X = np.loadtxt("iris.data", delimiter=",", usecols=range(4))
y_labels = np.loadtxt("iris.data", delimiter=",", usecols=[4], dtype="object")


label_to_int = dict(zip(np.unique(y), range(len(np.unique(y)))))

#shuffle the data:
indices = range(len(X))
random.shuffle(indices)
X = X[indices]
y = y[indices]

#split into train and test set:
X_train = X[0:100]
y_train = y[0:100]

X_test = X[100:]
y_test = y[100:]

#now we can fit a model:
experiment.fit(X_train, y_train)

#and predict the labels of the held out data:
y_predict = experiment.predict(X_test)

#Let's check what accuracy we get:


```


Advanced: Selecting specific classifiers
----------------------------------------

When you don't set a specific classifier all available classifiers will be tried. You have the option to limit the search to certain classifiers as follows:

First of all let's see what classifiers are available:

```python
import pyautoweka
print pyautoweka.AVAILABLE_CLASSIFIERS
```

Now let's say we want to just use the Simple Logistic classifier:
```python
experiment.add_classfier("weka.classifiers.functions.SimpleLogistic")
```


Advanced: files created
-----------------------

When you create a new experiment theres a bunch of files that will be generated before and during the run of AutoWeka. For each experiment there will be a new folder within in the `experiments` folder. The folder will have the name of the experiment, if it was specified in the constructor. Each time you fit data a tempraroy arff file will be created that holds all the data in it. This file will be delete after the `fit` call.

