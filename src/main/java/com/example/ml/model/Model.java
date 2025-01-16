package com.example.ml.model;

import com.example.ml.data.Instance;
import com.example.ml.evaluation.EvaluationMetrics;
import java.io.Serializable;
import java.util.List;

public interface Model<F extends Number, L extends Number> extends Serializable {
    void train(List<Instance<F, L>> trainSet, List<Instance<F, L>> validationSet);
    EvaluationMetrics test(List<Instance<F, L>> testSet);
}
