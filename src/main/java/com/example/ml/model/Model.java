package com.example.ml.model;

import com.example.ml.data.Instance;
import com.example.ml.evaluation.EvaluationMetrics;

import java.util.List;

public interface Model<F, L> {
    void train(List<Instance<F, L>> trainingSet, List<Instance<F, L>> validationSet);
    EvaluationMetrics test(List<Instance<F, L>> testSet);
}
