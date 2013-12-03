#Taken from http://www.debian-administration.org/article/An_introduction_to_bash_completion_part_2
_smac-validate()
{
   local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="--abortOnCrash --abortOnFirstRunCrash --algo --algoExec --checkInstanceFilesExist --configuration --cutoffLength --cutoffTime --cutoff_length --cutoff_time --deterministic --empericalPerformance --execDir --execdir --experimentDir --feature_file --instanceFeatureFile --instanceFile --instance_file --instance_seed_file --interInstanceObj --inter_instance_obj --intraInstanceObj --intra_instance_obj --leakMemory --leakMemoryAmount --logAllCallStrings --logAllProcessOutput --maxConcurrentAlgoExecs --maxTimestamp --minTimestamp --multFactor --numConcurrentAlgoExecs --numRun --numSeedsPerTestInstance --numTestInstances --numValidationRuns --numberOfConcurrentAlgoExecs --numberOfSeedsPerTestInstance --numberOfTestInstances --numberOfValidationRuns --outdir --outputDirectory --outputFileSuffix --overallObj --overall_obj --paramFile --paramfile --retryTargetAlgorithmRunCount --runHashCodeFile --runObj --run_obj --scenarioFile --seed --tae --taeSP --targetAlgorithmEvaluator --targetAlgorithmEvaluatorSearchPath --testInstanceFile --test_instance_file --test_instance_seed_file --trajectoryFile --tunerOverheadTime --tunerTime --tunerTimeout --useScenarioOutDir --validateOnlyLastIncumbent --validateTestInstances --validationHeaders --validationRoundingMode --verifySAT "

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _smac-validate smac-validate

#Taken from http://www.debian-administration.org/article/An_introduction_to_bash_completion_part_2
_smac()
{
   local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="--abortOnCrash --abortOnFirstRunCrash --adaptiveCapping --algo --algoExec --capAddSlack --capSlack --checkInstanceFilesExist --cleanOldStateOnSuccess --consoleLogLevel --countSMACTimeAsTunerTime --cutoffLength --cutoffTime --cutoff_length --cutoff_time --defaultConfigRuns --deterministic --doValidation --execDir --execdir --executionMode --expectedImprovementFunction --experimentDir --feature_file --frac_rawruntime --freeMemoryPecentageToSubsample --fullTreeBootstrap --help --ignoreConditionality --imputationIterations --initialChallenge --initialIncumbent --initialIncumbentRuns --initialN --instanceFeatureFile --instanceFile --instance_file --instance_seed_file --intensificationPercentage --interInstanceObj --inter_instance_obj --intraInstanceObj --intra_instance_obj --leakMemory --leakMemoryAmount --logAllCallStrings --logAllProcessOutput --logLevel --logModel --maskInactiveConditionalParametersAsDefaultValue --maxConcurrentAlgoExecs --maxIncumbentRuns --maxRunsForIncumbent --maxTimestamp --minTimestamp --minVariance --modelHashCodeFile --multFactor --nTrees --numChallengers --numConcurrentAlgoExecs --numEIRandomConfigs --numIterations --numPCA --numRandomConfigsInEI --numRun --numRunsLimit --numSeedsPerTestInstance --numTestInstances --numTrees --numValidationRuns --numberOfChallengers --numberOfConcurrentAlgoExecs --numberOfEIRandomConfigs --numberOfIterations --numberOfRandomConfigsInEI --numberOfRunsLimit --numberOfSeedsPerTestInstance --numberOfTestInstances --numberOfTrees --numberOfValidationRuns --optionFile --optionFile2 --outdir --outputDirectory --outputFileSuffix --overallObj --overall_obj --paramFile --paramfile --penalizeImputedValues --preprocessMarginal --ratioFeatures --restoreIteration --restoreStateFrom --restoreStateIteration --retryTargetAlgorithmRunCount --runGroupName --runHashCodeFile --runObj --run_obj --runtimeLimit --saveContext --saveContextWithState --scenarioFile --secondaryOptionsFile --seed --seedOffset --showHiddenParameters --shuffleImputedValues --splitMin --stateDeserializer --stateSerializer --storeDataInLeaves --subsamplePercentage --subsampleValuesWhenLowMemory --subsampleValuesWhenLowOnMemory --tae --taeSP --targetAlgorithmEvaluator --targetAlgorithmEvaluatorSearchPath --testInstanceFile --test_instance_file --test_instance_seed_file --totalNumRunsLimit --treatCensoredDataAsUncensored --tunerTimeout --useBrokenVarianceCalculation --validateOnlyLastIncumbent --validation --validationHeaders --validationRoundingMode --verifySAT --version --wallClockLimit "

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _smac smac

