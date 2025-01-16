package com.example.ml.evaluation;

import com.example.ml.data.Instance;

import java.util.List;

public class F1Score<F, L extends Number> implements EvaluationMeasure<F, L> {
    private L trueLabel;
    private L falseLabel;

    public F1Score(L trueLabel, L falseLabel) {
        this.trueLabel = trueLabel;
        this.falseLabel = falseLabel;
    }

    public double evaluate(List<Instance<F, L>> instances, List<L> predictions){

        Precision<F, L> precisionObj = new Precision<>(trueLabel, falseLabel);
        Recall<F, L> recallObj = new Recall<>(trueLabel, falseLabel);

        double precision = precisionObj.evaluate(instances, predictions);
        double recall = recallObj.evaluate(instances, predictions);

        return 2 * ((precision * recall) / (precision + recall));
    }
}
