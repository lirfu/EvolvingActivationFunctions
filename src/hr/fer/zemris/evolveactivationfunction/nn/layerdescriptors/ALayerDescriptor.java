package hr.fer.zemris.evolveactivationfunction.nn.layerdescriptors;

import hr.fer.zemris.utils.ISerializable;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

/**
 * A layer descriptor is used to easily build neural architectures by reading layer info from a string representation.
 */
public abstract class ALayerDescriptor implements ISerializable {
    protected String name_;

    public ALayerDescriptor(String name) {
        name_ = name;
    }

    public abstract Layer constructLayer();

    public String getName() {
        return name_;
    }

    public abstract ALayerDescriptor newInstance();
}
