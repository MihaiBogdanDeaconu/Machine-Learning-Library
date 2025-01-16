package com.example.ml.utils;

import com.example.ml.data.Instance;

import java.util.List;

public class SplitResult<F, L> {
    List<Instance<F, L>> trainSet;
    List<Instance<F, L>> validationSet;
    List<Instance<F, L>> testSet;

    public SplitResult(List<Instance<F, L>> trainSet, List<Instance<F, L>> validationSet, List<Instance<F, L>> testSet) {
        this.trainSet = trainSet;
        this.validationSet = validationSet;
        this.testSet = testSet;
    }

    public List<Instance<F, L>> getTrainSet() {
        return trainSet;
    }

    public List<Instance<F, L>> getValidationSet() {
        return validationSet;
    }

    public List<Instance<F, L>> getTestSet() {
        return testSet;
    }
}
