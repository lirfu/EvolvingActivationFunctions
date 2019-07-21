package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface IReport {
    public void build(String name, IModel model, Evaluation eval, ROCMultiClass roc, DataSetIterator it);
    public String toString();
}
