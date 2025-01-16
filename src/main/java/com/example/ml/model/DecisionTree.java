package com.example.ml.model;

import com.example.ml.data.Instance;
import com.example.ml.evaluation.*;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DecisionTree<F extends Number, L extends Number> implements Model<F, L> {

    private TreeNode root;
    private int maxDepth;
    private int minSamplesSplit;

    public DecisionTree(int maxDepth, int minSamplesSplit) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    @Override
    public void train(List<Instance<F, L>> trainSet, List<Instance<F, L>> validationSet) {
        this.root = buildTree(trainSet, 0);
        // Optionally, evaluate on validation set after training
        evaluateAndStore(validationSet, "Validation");
    }

    @Override
    public EvaluationMetrics test(List<Instance<F, L>> testSet) {
        List<L> predictions = getPredictions(testSet);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Precision<F, L> precision = new Precision<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Recall<F, L> recall = new Recall<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        F1Score<F, L> f1Score = new F1Score<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));

        double acc = accuracy.evaluate(testSet, predictions);
        double prec = precision.evaluate(testSet, predictions);
        double rec = recall.evaluate(testSet, predictions);
        double f1 = f1Score.evaluate(testSet, predictions);

        // Return the metrics instead of printing
        return new EvaluationMetrics(acc, prec, rec, f1);
    }

    private TreeNode buildTree(List<Instance<F, L>> data, int depth) {
        TreeNode node = new TreeNode();

        if (depth >= maxDepth || data.size() < minSamplesSplit || isPure(data)) {
            node.isLeaf = true;
            node.predictedLabel = majorityLabel(data);
            return node;
        }

        BestSplit bestSplit = findBestSplit(data);
        if (bestSplit == null) {
            node.isLeaf = true;
            node.predictedLabel = majorityLabel(data);
            return node;
        }

        node.splitFeatureIndex = bestSplit.featureIndex;
        node.splitThreshold = bestSplit.threshold;

        List<Instance<F, L>> leftData = new ArrayList<>();
        List<Instance<F, L>> rightData = new ArrayList<>();

        for (Instance<F, L> inst : data) {
            double value = inst.getFeatures().get(bestSplit.featureIndex).doubleValue();
            if (value <= bestSplit.threshold) {
                leftData.add(inst);
            } else {
                rightData.add(inst);
            }
        }

        node.leftChild = buildTree(leftData, depth + 1);
        node.rightChild = buildTree(rightData, depth + 1);

        return node;
    }

    private BestSplit findBestSplit(List<Instance<F, L>> data) {
        double bestImpurity = Double.POSITIVE_INFINITY;
        BestSplit bestSplit = null;

        int featureCount = data.get(0).getFeatures().size();
        for (int f = 0; f < featureCount; f++) {
            List<Double> values = new ArrayList<>();
            for (Instance<F, L> inst : data) {
                values.add(inst.getFeatures().get(f).doubleValue());
            }
            Collections.sort(values);

            for (int i = 0; i < values.size() - 1; i++) {
                if (values.get(i).equals(values.get(i + 1))) continue; // Skip identical values
                double threshold = (values.get(i) + values.get(i + 1)) / 2.0;
                double gini = computeSplitGini(data, f, threshold);
                if (gini < bestImpurity) {
                    bestImpurity = gini;
                    bestSplit = new BestSplit(f, threshold, gini);
                }
            }
        }
        return bestSplit;
    }

    private double computeSplitGini(List<Instance<F, L>> data, int featureIndex, double threshold) {
        List<Instance<F, L>> leftData = new ArrayList<>();
        List<Instance<F, L>> rightData = new ArrayList<>();
        for (Instance<F, L> inst : data) {
            double value = inst.getFeatures().get(featureIndex).doubleValue();
            if (value <= threshold) {
                leftData.add(inst);
            } else {
                rightData.add(inst);
            }
        }

        double leftGini = gini(leftData);
        double rightGini = gini(rightData);
        double leftWeight = (double) leftData.size() / data.size();
        double rightWeight = (double) rightData.size() / data.size();
        return leftWeight * leftGini + rightWeight * rightGini;
    }

    private double gini(List<Instance<F, L>> data) {
        if (data.isEmpty()) return 0.0;
        int count1 = 0; // label=1
        for (Instance<F, L> inst : data) {
            if (inst.getLabel().intValue() == 1) {
                count1++;
            }
        }
        double p1 = (double) count1 / data.size();
        double p0 = 1.0 - p1;
        return 1.0 - (p1 * p1 + p0 * p0);
    }

    private boolean isPure(List<Instance<F, L>> data) {
        if (data.isEmpty()) return true;
        int firstLabel = data.get(0).getLabel().intValue();
        for (Instance<F, L> inst : data) {
            if (inst.getLabel().intValue() != firstLabel) {
                return false;
            }
        }
        return true;
    }

    private int majorityLabel(List<Instance<F, L>> data) {
        int count1 = 0;
        for (Instance<F, L> inst : data) {
            if (inst.getLabel().intValue() == 1) {
                count1++;
            }
        }
        return (count1 >= data.size() - count1) ? 1 : 0;
    }

    public int predictSingle(Instance<F, L> instance) {
        TreeNode currentNode = root;
        while (!currentNode.isLeaf) {
            double value = instance.getFeatures().get(currentNode.splitFeatureIndex).doubleValue();
            if (value <= currentNode.splitThreshold) {
                currentNode = currentNode.leftChild;
            } else {
                currentNode = currentNode.rightChild;
            }
        }
        return currentNode.predictedLabel;
    }

    public List<L> getPredictions(List<Instance<F, L>> dataSet) {
        List<L> preds = new ArrayList<>();
        for (Instance<F, L> inst : dataSet) {
            preds.add((L) Integer.valueOf(predictSingle(inst)));
        }
        return preds;
    }

    private void evaluateAndStore(List<Instance<F, L>> dataset, String datasetName) {
        List<L> predictions = getPredictions(dataset);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Precision<F, L> precision = new Precision<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Recall<F, L> recall = new Recall<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        F1Score<F, L> f1Score = new F1Score<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));

        double acc = accuracy.evaluate(dataset, predictions);
        double prec = precision.evaluate(dataset, predictions);
        double rec = recall.evaluate(dataset, predictions);
        double f1 = f1Score.evaluate(dataset, predictions);

        System.out.println(datasetName + " Metrics:");
        System.out.println("Accuracy: " + String.format("%.4f", acc));
        System.out.println("Precision: " + String.format("%.4f", prec));
        System.out.println("Recall: " + String.format("%.4f", rec));
        System.out.println("F1 Score: " + String.format("%.4f", f1));
    }

    private List<L> convertPredictions(List<Integer> preds) {
        List<L> result = new ArrayList<>();
        for (Integer p : preds) {
            result.add((L) p);
        }
        return result;
    }

    private static class BestSplit {
        int featureIndex;
        double threshold;
        double impurity;

        public BestSplit(int featureIndex, double threshold, double impurity) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.impurity = impurity;
        }
    }

    // Plotting Validation Metrics
    public void plotMetrics(List<Double> accuracies, List<Double> precisions,
                            List<Double> recalls, List<Double> f1Scores) {
        List<Integer> iterations = new ArrayList<>();
        for (int i = 1; i <= accuracies.size(); i++) {
            iterations.add(i);
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Decision Tree Metrics")
                .xAxisTitle("Iteration")
                .yAxisTitle("Metric Value")
                .build();

        chart.addSeries("Accuracy", iterations, accuracies);
        chart.addSeries("Precision", iterations, precisions);
        chart.addSeries("Recall", iterations, recalls);
        chart.addSeries("F1 Score", iterations, f1Scores);

        new SwingWrapper<>(chart).displayChart();
    }
}
