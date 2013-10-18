import xml.etree.ElementTree as ET


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


class Experiment:

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

    def __init__(self):
        self.root = ET.Element('experimentBatch')
        self.tree = ET.ElementTree(self.root)

        self.set_experiment_parameters()

    def set_data_set(self, train_file, test_file=None, name="data"):
        """
        Add a dataset to the experiment.

        :param train_file: ARFF file containing the training data
        :param test_file: ARFF file containing the testing data, that will be
        used once the experiment completed (optional)
        :param name: name of the dataset (optional)
        """
        if self.root.find("datasetComponent"):
            self.root.remove(self.root.find("datasetComponent"))

        dataset_node = ET.SubElement(self.root, 'datasetComponent')
        train_file_node = ET.SubElement(dataset_node, 'trainArff')
        train_file_node.text = train_file
        test_file_node = ET.SubElement(dataset_node, 'testArff')
        if test_file:
            test_file_node.text = test_file
        else:
            #train_file not set, so use the train file again
            test_file_node.text = train_file
        name_node = ET.SubElement(dataset_node, 'name')
        name_node.text = name

    def set_experiment_parameters(
            self,
            name="Experiment",
            result_metric=RESULT_METRICS[0],
            optimization_method=OPTIMIZATION_METHOD[0],
            instance_generator=None,
            tuner_timeout=180,
            train_timeout=120,
            attribute_selection=True,
            attribute_selection_timeout=100
            ):
        """
        Set the parameters of the experiment.

        :param tuner_timeout: The number of seconds to run the SMBO method.
        :param train_timeout: The number of seconds to spend training a classifier with a set of hyperparameters
        on a given partition of the training set.
        """
        if result_metric not in Experiment.RESULT_METRICS:
            print ("FAILED: %s is not a valid result metric,"
                   " choose one from:" % result_metric)
            print ", ".join(Experiment.RESULT_METRICS)
            return

        if optimization_method not in Experiment.OPTIMIZATION_METHOD:
            print ("FAILED: %s is not a valid result metric,"
                   " choose one from:" % optimization_method)
            print ", ".join(Experiment.OPTIMIZATION_METHOD)
            return

        if (instance_generator
                and not isinstance(instance_generator, InstanceGenerator)):
            print "FAILED: instance_generator is not an InstanceGenerator"
            return

        if not isinstance(attribute_selection, bool):
            print "FAILED: attribute_selection needs to be a boolean"
            return

        #clear previous parameters:
        if self.root.find("selfComponent"):
            self.root.remove(self.root.find("selfComponent"))

        experiment = ET.SubElement(self.root, 'experimentComponent')
        name_node = ET.SubElement(experiment, 'name')
        name_node.text = name

        experiment_constructor = ET.SubElement(self.root,
                                               'experimentConstructor')
        experiment_constructor.text = Experiment.OPTIMIZATION_METHOD_CONSTRUCTOR[
                optimization_method]
        for experiment_arg in Experiment.OPTIMIZATION_METHOD_ARGS[
                optimization_method]:
            experiment_arg_node = ET.SubElement(experiment,
                                                'experimentConstructorArgs')
            experiment_arg_node.text = experiment_arg

        extra_props_node = ET.SubElement(experiment, 'extraProps')
        extra_props_node.text = Experiment.OPTIMIZATION_METHOD_EXTRA[
                optimization_method]

        instance_generator_node = ET.SubElement(experiment,
                                               'instanceGenerator')
        if not instance_generator:
            #Default generator
            instance_generator_node.text = "autoweka.instancegenerators.Default"
        else:
            instance_generator_node.text = instance_generator.name
            instance_generator_args_node = ET.SubElement(experiment,
                                                         'instanceGeneratorArgs')
            instance_generator_args_node.text = instance_generator.get_arg_str()

        tuner_timeout_node = ET.SubElement(experiment, 'tunerTimeout')
        tuner_timeout_node.text = str(tuner_timeout)
        train_timeout_node = ET.SubElement(experiment, 'trainTimeout')
        train_timeout_node.text = str(train_timeout)
            
        attribute_selection_node = ET.SubElement(experiment, 'attributeSelection')
        if attribute_selection:
            attribute_selection_node.text = "true"
            #TODO: attribute_selection_timeout 
        else:
            attribute_selection_node.text = "false"

        memory_node = ET.SubElement(experiment, 'memory')
        memory_node.text = "3000m"



    def write(self, file_name="experiment.xml"):
        self.tree.write(file_name)
