package hr.fer.zemris.evolveactivationfunction.nn;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IOpenLayer {
    public INDArray getActivation();

    public void setMeasuring(boolean measuring);
}
