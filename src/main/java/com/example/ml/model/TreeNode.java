package com.example.ml.model;

import java.util.List;

public class TreeNode {
    // Splitting info
    public int splitFeatureIndex = -1;
    public double splitThreshold = Double.NaN;

    // Children
    public TreeNode leftChild;
    public TreeNode rightChild;

    // Leaf info
    public boolean isLeaf = false;
    public int predictedLabel = -1; // or 0/1

    // Constructor
    public TreeNode() {}
}
