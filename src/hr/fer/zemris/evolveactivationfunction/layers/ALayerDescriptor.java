package hr.fer.zemris.evolveactivationfunction.layers;

import hr.fer.zemris.utils.ISerializable;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

public abstract class ALayerDescriptor implements ISerializable {
    protected String name_;

    public ALayerDescriptor(String name) {
        name_ = name;
    }

    public abstract Layer constructLayer(IActivation activation);

    public String getName() {
        return name_;
    }

    public abstract ALayerDescriptor newInstance();
}
