package com.example.ml.evaluation;

public class EvaluationMetrics {
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;

    public EvaluationMetrics(double accuracy, double precision, double recall, double f1Score) {
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
    }

    // Getters
    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
}
