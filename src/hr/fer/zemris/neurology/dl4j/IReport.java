package hr.fer.zemris.neurology.dl4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;

public interface IReport {
    public void build(ModelParams params, MultiLayerNetwork network, Evaluation eval, ROCMultiClass roc);
}
