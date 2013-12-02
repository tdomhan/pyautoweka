pyautoweka
==========

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





