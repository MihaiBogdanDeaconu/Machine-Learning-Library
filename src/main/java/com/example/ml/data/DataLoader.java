package com.example.ml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class DataLoader<F, L> {

    private final Function<String, F> featureParser;
    private final Function<String, L> labelParser;

    public DataLoader(Function<String, F> featureParser, Function<String, L> labelParser) {
        this.featureParser = featureParser;
        this.labelParser = labelParser;
    }

    public List<Instance<F, L>> loadFromCsv(String filePath){
        List<Instance<F,L>> instances = new ArrayList<>();
        boolean header = true;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!header) {
                    String[] values = line.split(",");
                    List<F> features = new ArrayList<>();
                    for (int i = 0; i < values.length - 1; i++) {
                        F feature = featureParser.apply(values[i]);
                        features.add(feature);
                    }
                    L label = labelParser.apply(values[values.length - 1]);

                    instances.add(new Instance<>(features, label));
                }
               else {
                   header = false;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return instances;
    }
}

