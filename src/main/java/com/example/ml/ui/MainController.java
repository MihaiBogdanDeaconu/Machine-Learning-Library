package com.example.ml.ui;

import com.example.ml.data.DataLoader;
import com.example.ml.data.Dataset;
import com.example.ml.data.Instance;
import com.example.ml.evaluation.EvaluationMetrics;
import com.example.ml.model.*;
import com.example.ml.utils.ModelSerializer;
import com.example.ml.utils.SplitResult;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MainController {

    private static final Logger LOGGER = Logger.getLogger(MainController.class.getName());

    @FXML
    private TextField filePathField;

    @FXML
    private ComboBox<String> classifierComboBox;

    @FXML
    private VBox hyperparametersBox;

    @FXML
    private TextField trainSplitField;

    @FXML
    private TextField testSplitField;

    @FXML
    private Label accuracyLabel;

    @FXML
    private Label precisionLabel;

    @FXML
    private Label recallLabel;

    @FXML
    private Label f1ScoreLabel;

    // Hyperparameter fields
    private TextField learningRateField;
    private TextField maxEpochsField;
    private TextField batchSizeField;
    private TextField maxDepthField;
    private TextField minSamplesSplitField;

    // Current trained model
    private Model<Double, Integer> trainedModel;

    @FXML
    public void initialize() {
        // Initialize classifier options programmatically
        classifierComboBox.getItems().addAll("Perceptron", "Logistic Regression", "Decision Tree");
        classifierComboBox.getSelectionModel().selectFirst();
        handleClassifierSelection(null);
    }

    @FXML
    private void handleBrowse(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Input CSV File");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File selectedFile = fileChooser.showOpenDialog(filePathField.getScene().getWindow());
        if (selectedFile != null) {
            filePathField.setText(selectedFile.getAbsolutePath());
        }
    }

    @FXML
    private void handleClassifierSelection(ActionEvent event) {
        hyperparametersBox.getChildren().clear();
        String selectedClassifier = classifierComboBox.getValue();

        switch (selectedClassifier) {
            case "Perceptron":
                addHyperparameterField("Learning Rate:", "learningRateField", "0.01");
                addHyperparameterField("Max Epochs:", "maxEpochsField", "50");
                addHyperparameterField("Batch Size:", "batchSizeField", "1");
                break;
            case "Logistic Regression":
                addHyperparameterField("Learning Rate:", "learningRateField", "0.01");
                addHyperparameterField("Max Epochs:", "maxEpochsField", "50");
                addHyperparameterField("Batch Size:", "batchSizeField", "16");
                break;
            case "Decision Tree":
                addHyperparameterField("Max Depth:", "maxDepthField", "7");
                addHyperparameterField("Min Samples Split:", "minSamplesSplitField", "20");
                break;
            default:
                break;
        }
    }

    private void addHyperparameterField(String labelText, String fxId, String defaultValue) {
        Label label = new Label(labelText);
        TextField textField = new TextField(defaultValue);
        textField.setId(fxId);
        hyperparametersBox.getChildren().addAll(label, textField);
    }

    @FXML
    private void handleTrain(ActionEvent event) {
        String filePath = filePathField.getText();
        if (filePath.isEmpty()) {
            showAlert(Alert.AlertType.ERROR, "Error", "Please select an input CSV file.");
            return;
        }

        String selectedClassifier = classifierComboBox.getValue();

        double trainRatio;
        double testRatio;

        try {
            trainRatio = Double.parseDouble(trainSplitField.getText()) / 100.0;
            testRatio = Double.parseDouble(testSplitField.getText()) / 100.0;

            if (trainRatio + testRatio > 1.0) {
                showAlert(Alert.AlertType.ERROR, "Error", "Train and Test splits exceed 100%.");
                return;
            }
        } catch (NumberFormatException e) {
            showAlert(Alert.AlertType.ERROR, "Error", "Invalid train-test split percentages.");
            return;
        }

        // Disable UI elements during training
        setUIEnabled(false);

        Task<Void> trainTask = new Task<>() {
            @Override
            protected Void call() {
                try {
                    // Load Data
                    DataLoader<Double, Integer> loader = new DataLoader<>(
                            Double::parseDouble,
                            Integer::parseInt
                    );
                    List<Instance<Double, Integer>> data = loader.loadFromCsv(filePath);

                    // Initialize Dataset
                    Dataset<Double, Integer> dataset = new Dataset<>(data, 8, Double::parseDouble);
                    dataset.shuffle();

                    // Initialize variables
                    Model<Double, Integer> model;
                    SplitResult<Double, Integer> splitResult;

                    // Variable to hold metrics
                    EvaluationMetrics metrics;

                    switch (selectedClassifier) {
                        case "Perceptron":
                            // Fetch hyperparameters
                            learningRateField = (TextField) hyperparametersBox.lookup("#learningRateField");
                            maxEpochsField = (TextField) hyperparametersBox.lookup("#maxEpochsField");
                            batchSizeField = (TextField) hyperparametersBox.lookup("#batchSizeField");

                            double perceptronLearningRate = Double.parseDouble(learningRateField.getText());
                            int perceptronMaxEpochs = Integer.parseInt(maxEpochsField.getText());
                            int perceptronBatchSize = Integer.parseInt(batchSizeField.getText());

                            // Convert labels from 0/1 to -1/1
                            dataset.convertLabelsToMinusOne();

                            // Split data with standardization
                            splitResult = dataset.trainTestSplit(trainRatio, true);
                            List<Instance<Double, Integer>> trainSet = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSet = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSet = splitResult.getTestSet();

                            // Initialize and train Perceptron
                            model = new Perceptron<>(8, perceptronLearningRate, perceptronMaxEpochs, perceptronBatchSize);
                            model.train(trainSet, validationSet);

                            // Assign to trainedModel for serialization
                            trainedModel = model;

                            // Test and get metrics
                            metrics = model.test(testSet);
                            break;

                        case "Logistic Regression":
                            // Fetch hyperparameters
                            learningRateField = (TextField) hyperparametersBox.lookup("#learningRateField");
                            maxEpochsField = (TextField) hyperparametersBox.lookup("#maxEpochsField");
                            batchSizeField = (TextField) hyperparametersBox.lookup("#batchSizeField");

                            double lrLearningRate = Double.parseDouble(learningRateField.getText());
                            int lrMaxEpochs = Integer.parseInt(maxEpochsField.getText());
                            int lrBatchSize = Integer.parseInt(batchSizeField.getText());

                            // Split data with standardization
                            splitResult = dataset.trainTestSplit(trainRatio, true);
                            List<Instance<Double, Integer>> trainSetLR = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSetLR = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSetLR = splitResult.getTestSet();

                            // Initialize and train Logistic Regression
                            model = new LogisticRegression<>(8, lrLearningRate, lrMaxEpochs, lrBatchSize);
                            model.train(trainSetLR, validationSetLR);

                            // Assign to trainedModel for serialization
                            trainedModel = model;

                            // Test and get metrics
                            metrics = model.test(testSetLR);
                            break;

                        case "Decision Tree":
                            // Fetch hyperparameters
                            maxDepthField = (TextField) hyperparametersBox.lookup("#maxDepthField");
                            minSamplesSplitField = (TextField) hyperparametersBox.lookup("#minSamplesSplitField");

                            int dtMaxDepth = Integer.parseInt(maxDepthField.getText());
                            int dtMinSamplesSplit = Integer.parseInt(minSamplesSplitField.getText());

                            // Split data without standardization
                            splitResult = dataset.trainTestSplit(trainRatio, false);
                            List<Instance<Double, Integer>> trainSetDT = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSetDT = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSetDT = splitResult.getTestSet();

                            // Initialize and train Decision Tree
                            model = new DecisionTree<>(dtMaxDepth, dtMinSamplesSplit);
                            model.train(trainSetDT, validationSetDT);

                            // Assign to trainedModel for serialization
                            trainedModel = model;

                            // Test and get metrics
                            metrics = model.test(testSetDT);
                            break;

                        default:
                            throw new IllegalArgumentException("Unsupported classifier selected.");
                    }

                    double finalAccuracy = metrics.getAccuracy() * 100.0;
                    double finalPrecision = metrics.getPrecision() * 100.0;
                    double finalRecall = metrics.getRecall() * 100.0;
                    double finalF1Score = metrics.getF1Score() * 100.0;

                    Platform.runLater(() -> {
                        accuracyLabel.setText(String.format("%.2f%%", finalAccuracy));
                        precisionLabel.setText(String.format("%.2f%%", finalPrecision));
                        recallLabel.setText(String.format("%.2f%%", finalRecall));
                        f1ScoreLabel.setText(String.format("%.2f%%", finalF1Score));

                        setUIEnabled(true);
                        showAlert(Alert.AlertType.INFORMATION, "Success", "Training and Evaluation Completed.");
                    });

                } catch (Exception e) {
                    LOGGER.log(Level.SEVERE, "Training failed", e);
                    Platform.runLater(() -> {
                        setUIEnabled(true);
                        showAlert(Alert.AlertType.ERROR, "Error", "Training failed: " + e.getMessage());
                    });
                }
                return null;
            }
        };

        new Thread(trainTask).start();
    }

    @FXML
    private void handleSaveModel(ActionEvent event) {
        if (trainedModel == null) {
            showAlert(Alert.AlertType.ERROR, "Error", "No trained model to save. Please train a model first.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Trained Model");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Model Files", "*.model"));
        File file = fileChooser.showSaveDialog(filePathField.getScene().getWindow());
        if (file != null) {
            try {
                ModelSerializer.saveModel(trainedModel, file.getAbsolutePath());
                showAlert(Alert.AlertType.INFORMATION, "Success", "Model saved successfully.");
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "Failed to save model", e);
                showAlert(Alert.AlertType.ERROR, "Error", "Failed to save model: " + e.getMessage());
            }
        }
    }

    @FXML
    private void handleLoadModel(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Load Trained Model");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Model Files", "*.model"));
        File file = fileChooser.showOpenDialog(filePathField.getScene().getWindow());
        if (file != null) {
            try {
                trainedModel = ModelSerializer.loadModel(file.getAbsolutePath());
                showAlert(Alert.AlertType.INFORMATION, "Success", "Model loaded successfully.");
                // Optionally, you can add code to display model details or evaluate on test data
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "Failed to load model", e);
                showAlert(Alert.AlertType.ERROR, "Error", "Failed to load model: " + e.getMessage());
            }
        }
    }

    private void setUIEnabled(boolean enabled) {
        filePathField.setDisable(!enabled);
        classifierComboBox.setDisable(!enabled);
        hyperparametersBox.setDisable(!enabled);
        trainSplitField.setDisable(!enabled);
        testSplitField.setDisable(!enabled);
    }

    private void showAlert(Alert.AlertType type, String title, String message) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
