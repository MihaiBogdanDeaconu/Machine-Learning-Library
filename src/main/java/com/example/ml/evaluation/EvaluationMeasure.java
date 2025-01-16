package com.example.ml.evaluation;

import com.example.ml.data.Instance;

import java.util.List;

public interface EvaluationMeasure<F, L> {
    double evaluate(List<Instance<F, L>> instances, List<L> predictions);
}
