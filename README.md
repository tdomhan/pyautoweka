pyautoweka
==========


Installation
------------

Go to the project sources and run:
```
python setup.py install
```

Running a experiment
--------------------

AutoWeka for python.

```python
import pyautoweka

#Create an experiment
experiment = pyautoweka.Experiment()

#Add some data to it
experiment.add_data_set("./datasets/creditg.arff")

#Run the experiment
experiment.run()
```


Advanced: Selecting specific classifiers
----------------------------------------

When you don't set a specific classifier all available classifiers will be tried. You have the option to limit the search to certain classifiers as follows:

First of all let's see what classifiers are available:
```
import pyautoweka
print pyautoweka.AVAILABLE_CLASSIFIERS
```

Now let's say we want to just use the Simple Logistic classifier:
```
experiment.add_classfier("weka.classifiers.functions.SimpleLogistic")
```

