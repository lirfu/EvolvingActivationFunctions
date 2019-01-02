package hr.fer.zemris.evolveactivationfunction.activationfunction;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

public class CustomFunction extends BaseActivationFunction {
    private DerivableSymbolicTree tree_;

    public CustomFunction(DerivableSymbolicTree tree) {
        tree_ = tree;
    }

    @Override
    public INDArray getActivation(INDArray input, boolean b) {
        return tree_.execute(input);
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray input, INDArray epsilon) {
        INDArray dLds = tree_.derivate(input);
        dLds.muli(epsilon);
        return new Pair<>(dLds, epsilon);
    }
}
