package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public interface IModel {
    public void setModel(MultiLayerNetwork m);
}
