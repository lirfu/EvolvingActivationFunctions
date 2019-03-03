package hr.fer.zemris.evolveactivationfunction.layers;

import hr.fer.zemris.utils.ISerializable;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

public abstract class ALayerDescriptor implements ISerializable {
    protected String name_;

    public ALayerDescriptor(String name) {
        name_ = name;
    }

    public abstract Layer constructLayer(int input_num, IActivation activation);

    public abstract String getName();

    public abstract int outputNum();

    public abstract ALayerDescriptor newInstance();
}
