package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;

public interface IReport {
    public void build(String name, IModel network, Evaluation eval, ROCMultiClass roc);
    public String toString();
}
