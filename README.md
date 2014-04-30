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
`tuner_timeout` is the time the optimization will run in seconds. So e.g. 9000 seconds = 2.5 hours. The longer you run the optimization, the better of course.

```python
#Either set the path to a dataset in the arff format:
experiment.set_data_set_files("./dataset.arff")
#Or provide a numpy ndarray:
experiment.set_data_set(X,y)

#Run the experiment:
experiment.run()

#Make predictions:
experiment.predict_from_file(data_file="testdataset.arff")
#Or from a numpy ndarray:
experiment.predict(X)
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

