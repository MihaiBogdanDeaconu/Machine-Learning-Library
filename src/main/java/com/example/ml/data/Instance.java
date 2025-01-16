package com.example.ml.data;

import java.util.List;
import java.util.function.Function;

public class Instance<F, L> {
    List<F> features;
    L label;

    public Instance(List<F> features, L label) {
        this.features = features;
        this.label = label;
    }

    public List<F> getFeatures() {
        return features;
    }

    public void setFeature(F feature, int index) {
        this.features.set(index, feature);
    }

    public L getLabel() {
        return label;
    }

    public void setLabel(L label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return "Instance{" +
                "features=" + features +
                ", label=" + label +
                '}';
    }
}
