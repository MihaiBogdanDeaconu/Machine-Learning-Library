package com.example.ml.evaluation;

import com.example.ml.data.Instance;

import java.util.List;

public class Precision<F, L extends Number> implements EvaluationMeasure<F, L> {
    private L trueLabel;
    private L falseLabel;

    public Precision(L trueLabel, L falseLabel) {
        this.trueLabel = trueLabel;
        this.falseLabel = falseLabel;
    }

    public double evaluate(List<Instance<F, L>> instances, List<L> predictions){
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;

        for(int i = 0; i < instances.size(); i++) {
            if(instances.get(i).getLabel() == trueLabel){
                if(predictions.get(i) == trueLabel){
                    tp++;
                }
                else{
                    fn++;
                }
            }
            else if(instances.get(i).getLabel() == falseLabel){
                if(predictions.get(i) == falseLabel){
                    tn++;
                }
                else{
                    fp++;
                }
            }
        }

        return ((double) tp) / ((double) tp + (double) fp);
    }
}
