import xml.etree.ElementTree as ET
import xml.dom.minidom
import os


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
        :param num_folds: The number of folds to generate
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
            "-experimentpath", "experiments", "-propertyoverride",
            "smacexecutable=smac-v2.04.01-master-447-patched/smac"
            ],
        "TPE": [
            "-experimentpath", "experiments", "-propertyoverride",
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

    def write_xml(self, file_name="experiment.xml"):
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

        tree.write(file_name)
        print xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml()


    def add_data_set(self, train_file, test_file=None, name="data"):
        """
        Add a dataset to the experiment.

        :param train_file: ARFF file containing the training data
        :param test_file: ARFF file containing the testing data, that will be
        used once the experiment completed (optional)
        :param name: name of the dataset (optional)
        """
        self.datasets.append(DataSet(train_file, test_file, name))
