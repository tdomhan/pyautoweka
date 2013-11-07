import xml.etree.ElementTree as ET
import xml.dom.minidom
from subprocess import call
import datetime
import os
import imp

"""
NOTES

 * auto-wekalight doesn't work: 
            java -cp "./lib/weka.jar;autoweka-light.jar" autoweka.ExperimentConstructor experiments/experiment.xml
            Error: Could not find or load main class autoweka.ExperimentConstructor

 * How do you get available classifiers??
    * read all param files? or interface pyautoweka somehow

 * Get the directory of the python package:
    import imp
    e.g. imp.find_module("sklearn")[1]

 * creating the actual experiment from the XML file:
   java -cp autoweka.jar autoweka.ExperimentConstructor experiments/experiment.xml

 * running the experiment afterwards:
     arguments: folder + seed
   java -cp autoweka.jar autoweka.Experiment experiments/Experiment-data 0

"""


EXPERIMENT_FOLDER = "experiments"

PYAUTOWEKA_BASEDIR = imp.find_module("pyautoweka")[1]
#TODO: fix to current dir until we actually install it
PYAUTOWEKA_BASEDIR = "./"

def get_available_classifiers(base_dir):
    """
        Determine the available classifiers by iterating over
        all parameter files.
    """
    params_dir = os.path.join(PYAUTOWEKA_BASEDIR,"params")
    classifiers = []
    for root, dir, files in os.walk(params_dir):
        for file in files:
            if file.startswith("weka.classifiers") and file.endswith(".params"):
                clf = file[0:-len(".params")]
                classifiers.append(clf)
    return classifiers

AVAILABLE_CLASSIFIERS = get_available_classifiers(PYAUTOWEKA_BASEDIR)

class InstanceGenerator(object):
    def __init__(self):
        self.name = "Default"
        self.params = {}

    def get_arg_str(self):
        key_value_str = lambda key, value: "%s=%s" % (str(key), str(value))
        return ":".join([key_value_str(key, value)
                         for key, value in self.params.iteritems()])


class CrossValidation(InstanceGenerator):
    """
    Performs k-fold cross validation on the training set.
    """
    def __init__(self, seed=0, num_folds=10):
        """
        :param seed: The seed to use for randomizing the dataset
        :param num_fold./s: The number of folds to generate
        """
        super(CrossValidation, self).__init__()
        self.name = "autoweka.instancegenerators.CrossValidation"
        self.params["seed"] = seed
        self.params["numFolds"] = num_folds


class RandomSubSampling(InstanceGenerator):
    """
    Performs generates an arbitrary number of folds by randomly
    making a partition of the training data of a fixed percentage.
    """
    def __init__(self, starting_seed=0, num_samples=10,
                 percent_training=70, bias_to_uniform=None):
        """

        :param starting_seed: The seed to use for randomizing the dataset
        :param num_samples: The number of subsamples to generate
        :param percent_training: The percent of the training data to use
        as 'new training data'
        :param bias_to_uniform: The bias towards a uniform class
        distribution (optional)
        """
        super(RandomSubSampling, self).__init__()
        self.name = "autoweka.instancegenerators.RandomSubSampling"
        self.params["startingSeed"] = starting_seed
        self.params["numSamples"] = num_samples
        self.params["percent"] = percent_training
        if bias_to_uniform:
            self.params["bias"] = bias_to_uniform


class DataSet(object):
    def __init__(self, train_file, test_file=None, name="data"):
        """
        Dataset.

        :param train_file: ARFF file containing the training data
        :param test_file: ARFF file containing the testing data, that will be
        used once the experiment completed (optional)
        :param name: name of the dataset (optional)
        """
        self.train_file = os.path.abspath(train_file)
        if test_file:
            self.test_file = os.path.abspath(test_file)
        else:
            self.test_file = None
        self.name = name


class Experiment:
    """
      TODO: classifier selection!
    """

    RESULT_METRICS = ["errorRate",
                      "rmse",
                      "rrse",
                      "meanAbsoluteErrorMetric",
                      "relativeAbsoluteErrorMetric"]

    OPTIMIZATION_METHOD = ["SMAC", "TPE"]

    OPTIMIZATION_METHOD_CONSTRUCTOR = {
        "SMAC": "autoweka.smac.SMACExperimentConstructor",
        "TPE":  "autoweka.tpe.TPEExperimentConstructor"}

    OPTIMIZATION_METHOD_ARGS = {
        "SMAC": [
            "-experimentpath", os.path.abspath(EXPERIMENT_FOLDER),
            "-propertyoverride",
            ("smacexecutable=%s" 
             "smac-v2.04.01-master-447-patched/smac" % PYAUTOWEKA_BASEDIR)
            ],
        "TPE": [
            "-experimentpath", os.path.abspath(EXPERIMENT_FOLDER),
            "-propertyoverride",
            ("pythonpath=$PYTHONPATH\:~/src/hyperopt\:~/src/hyperopt/external:"
             "tperunner=./src/python/tperunner.py:python=/usr/bin/python2")
            ]
        }

    OPTIMIZATION_METHOD_EXTRA = {
        "SMAC": "executionMode=SMAC:initialIncumbent=RANDOM:initialN=1",
        "TPE": ""
        }

    def __init__(
            self,
            experiment_name="Experiment",
            result_metric=RESULT_METRICS[0],
            optimization_method=OPTIMIZATION_METHOD[0],
            instance_generator=None,
            tuner_timeout=180,
            train_timeout=120,
            attribute_selection=True,
            attribute_selection_timeout=100,
            memory="3000m"
            ):
        """
        Create a new experiment.

        :param tuner_timeout: The number of seconds to run the SMBO method.
        :param train_timeout: The number of seconds to spend training
        a classifier with a set of hyperparameters on a given partition of
        the training set.
        """
        if result_metric not in Experiment.RESULT_METRICS:
            raise ValueError("%s is not a valid result metric,"
                             " choose one from: %s" % (
                                 result_metric,
                                 ", ".join(Experiment.RESULT_METRICS)))

        if optimization_method not in Experiment.OPTIMIZATION_METHOD:
            raise ValueError("%s is not a valid optimization method,"
                             " choose one from:" % (
                                 optimization_method,
                                 ", ".join(Experiment.OPTIMIZATION_METHOD)))

        if (instance_generator
                and not isinstance(instance_generator, InstanceGenerator)):
            raise ValueError(("instance_generator needs to be"
                              " an InstanceGenerator or None"))

        if not isinstance(attribute_selection, bool):
            raise ValueError("attribute_selection needs to be a boolean")

        self.experiment_name = experiment_name
        self.result_metric = result_metric
        self.optimization_method = optimization_method
        self.instance_generator = instance_generator
        self.tuner_timeout = tuner_timeout
        self.train_timeout = train_timeout
        self.attribute_selection = attribute_selection
        self.attribute_selection_timeout = attribute_selection_timeout
        self.memory = memory

        self.datasets = []
        self.classifiers = []

        self.file_name = None

        self.prepared = False

    def _get_xml(self):
        """
        Write this experiment as a valid xml that can be read by Auto-WEKA.
        """

        root = ET.Element('experimentBatch')
        tree = ET.ElementTree(root)

        experiment = ET.SubElement(root, 'experimentComponent')

        name_node = ET.SubElement(experiment, 'name')
        name_node.text = self.experiment_name

        result_metric_node = ET.SubElement(experiment, 'resultMetric')
        result_metric_node.text = self.result_metric

        experiment_constructor = ET.SubElement(experiment,
                                               'experimentConstructor')
        experiment_constructor.text = Experiment.OPTIMIZATION_METHOD_CONSTRUCTOR[
            self.optimization_method]
        for experiment_arg in Experiment.OPTIMIZATION_METHOD_ARGS[
                self.optimization_method]:
            experiment_arg_node = ET.SubElement(experiment,
                                                'experimentConstructorArgs')
            experiment_arg_node.text = experiment_arg

        extra_props_node = ET.SubElement(experiment, 'extraProps')
        extra_props_node.text = Experiment.OPTIMIZATION_METHOD_EXTRA[
            self.optimization_method]

        instance_generator_node = ET.SubElement(experiment,
                                                'instanceGenerator')
        if not self.instance_generator:
            #Default generator
            instance_generator_node.text = "autoweka.instancegenerators.Default"
            instance_generator_args_node = ET.SubElement(
                experiment,
                'instanceGeneratorArgs')
            instance_generator_args_node.text = ""
        else:
            instance_generator_node.text = self.instance_generator.name
            instance_generator_args_node = ET.SubElement(
                experiment,
                'instanceGeneratorArgs')
            instance_generator_args_node.text = self.instance_generator.get_arg_str()

        tuner_timeout_node = ET.SubElement(experiment, 'tunerTimeout')
        tuner_timeout_node.text = str(self.tuner_timeout)
        train_timeout_node = ET.SubElement(experiment, 'trainTimeout')
        train_timeout_node.text = str(self.train_timeout)

        attribute_selection_node = ET.SubElement(experiment, 'attributeSelection')
        if self.attribute_selection:
            attribute_selection_node.text = "true"
            attr_select_timeout_node = ET.SubElement(
                experiment, 'attributeSelectionTimeout')
            attr_select_timeout_node.text = str(self.attribute_selection_timeout)
        else:
            attribute_selection_node.text = "false"

        for classifier in self.classifiers:
            classifier_node = ET.SubElement(experiment, 'allowedClassifiers')
            classifier_node.text = classifier

        memory_node = ET.SubElement(experiment, 'memory')
        memory_node.text = str(self.memory)

        # Write all dataset components:

        for dataset in self.datasets:
            dataset_node = ET.SubElement(root, 'datasetComponent')
            train_file_node = ET.SubElement(dataset_node, 'trainArff')
            train_file_node.text = dataset.train_file
            test_file_node = ET.SubElement(dataset_node, 'testArff')
            if dataset.test_file:
                test_file_node.text = dataset.test_file
            else:
                #train_file not set, so use the train file again
                test_file_node.text = dataset.train_file
            name_node = ET.SubElement(dataset_node, 'name')
            name_node.text = dataset.name

        return tree

    def __repr__(self):
        root = self._get_xml().getroot()
        return xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml()

    def _write_xml(self, file_name="experiment.xml"):
        tree = self._get_xml()
        self.file_name = file_name
        tree.write(file_name)

    def add_data_set(self, train_file, test_file=None, name=None):
        """
        Add a dataset to the experiment.

        :param train_file: ARFF file containing the training data
        :param test_file: ARFF file containing the testing data, that will be
        used once the experiment completed (optional)
        :param name: name of the dataset (optional)
        """
        if not os.path.exists(train_file):
            raise Exception("train_file doesn't exist")
        if test_file is not None and not os.path.exists(test_file):
            raise Exception("test_file doesn't exist")
        if name == None:
            name = os.path.basename(train_file)
        #check there's not other dataset with the same name
        for dataset in self.datasets:
            if dataset.name == name:
                raise ValueError("A dataset with the name '%s', was already added." % name)
        self.datasets.append(DataSet(train_file, test_file, name))

    def add_classfier(self, clf):
        """
        Restrict the search to a certain classifier. Call multiple times to select more than one.
        If not called, all classifiers will be used.

        For a list of available classifiers see: pyautoweka.AVAILABLE_CLASSIFIERS

        :param clf: the classifier
        """
        if not clf in AVAILABLE_CLASSIFIERS:
            raise ValueError("%s is not one of the AVAILABLE_CLASSIFIERS." % clf)
        self.classifiers.append(clf)

    def prepare(self):
        """
        Creates the experiment folder.

        java -cp autoweka.jar autoweka.ExperimentConstructor
        """
        if len(self.datasets) == 0:
            raise Exception("No datasets added yet, see Experiment.add_data_set")
        self._write_xml(self.experiment_name + ".xml")
        experiment_constructor = [ "java",
                                   "-cp",
                                   "autoweka.jar",
                                   "autoweka.ExperimentConstructor",
                                   self.file_name]
        ret = call(experiment_constructor)
        if ret == 0:
            #TODO: check return type for errors
            self.prepared = True
            return
        else:
            self.prepared = False
            raise Exception("Could not prepare the experiment")

    def run(self, seed=0, hide_output=True):
        """
            Run a experiment that was previously created

            :param seed: seed for the random number generator
        """
        if not self.prepared:
            self.prepare()
        print "Running experiments"
        print "Time allocated(see Experiment.tuner_timeout): ", str(datetime.timedelta(seconds=self.tuner_timeout))
        for dataset in self.datasets:
            print "Running experiment on dataset %s" % dataset.name
            experiment_folder = os.path.join(EXPERIMENT_FOLDER,
                                             self.experiment_name + "-" + dataset.name)
            experiment_runner = [ "java",
                                  "-cp",
                                  "autoweka.jar",
                                  "autoweka.tools.ExperimentRunner",
                                  experiment_folder,
                                  str(seed)]
            print " ".join(experiment_runner)
            if hide_output:
                call(experiment_runner,
                     stdout=open(os.devnull),
                     stderr=open(os.devnull))
            else:
                call(experiment_runner)


