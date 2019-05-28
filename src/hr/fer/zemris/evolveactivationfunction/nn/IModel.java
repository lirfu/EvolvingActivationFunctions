package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;

public interface IModel {
    public void setModel(MultiLayerNetwork m);

    public List<Double> getTrainLosses();

    public List<Double> getTestLosses();
}
