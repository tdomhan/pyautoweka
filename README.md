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

First we need to load some data. Let's for example the the famous [Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris). Download it using [this link](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

```python
```

```python
experiment.fit(X, y)
y_predict = experimebt.predict(X)
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

