package hr.fer.zemris.neurology.demo;

import hr.fer.zemris.data.datasets.BinaryDecoderClassification;
import hr.fer.zemris.neurology.dl4j.CommonModel;
import hr.fer.zemris.neurology.dl4j.Dl4jDataset;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.FileNotFoundException;
import java.io.IOException;

public class DemoDL4J {

    public static void main(String[] args) throws IOException {
//        originalTest();
        dataFromGeneratorTest();
    }

    private static class MyActivation extends BaseActivationFunction {
        @Override
        public INDArray getActivation(INDArray input, boolean training_mode) {
            Nd4j.getExecutioner().execAndReturn(new Sigmoid(input));
            return input;
        }

        @Override
        public Pair<INDArray, INDArray> backprop(INDArray input, INDArray gradient) {
            // calculate the function gradient in given input
            INDArray out = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(input));
            // multiply with incoming gradient (dL/do)
            out.muli(gradient);
            // gradients toward inputs and the act. func. inner parameters (or null if func. has no learnable params)
            return new Pair<>(out, null);
        }
    }

    public static void originalTest() throws IOException {
        CommonModel.Params p = new CommonModel.Params(
                28 * 28, 10, 10, 64, 0.0015, 0.0015 * 0.005, 42
        );
        DataSetIterator mnistTrain = new MnistDataSetIterator(p.batch_size(), true, (int) p.seed());
        DataSetIterator mnistTest = new MnistDataSetIterator(p.batch_size(), false, (int) p.seed());

        CommonModel model = new CommonModel(p, new int[]{500, 100}, new IActivation[]{new MyActivation()});
        StdoutLogger log = new StdoutLogger();

        model.train(mnistTrain, log);
        model.test(mnistTest, log);
    }

    public static void dataFromGeneratorTest() throws FileNotFoundException {
        DataSetIterator dataset = new Dl4jDataset(new BinaryDecoderClassification(), 3, 42);
        CommonModel.Params p = new CommonModel.Params(dataset.inputColumns(), dataset.totalOutcomes(),
                500, dataset.batch(), 1e-1, 1e-9, 42);

        CommonModel model = new CommonModel(p, new int[]{5, 5}, new IActivation[]{new MyActivation()});
        StdoutLogger log = new StdoutLogger();

        dataset.reset();
        model.train(dataset, log);
        dataset.reset();
        model.test(dataset, log);
    }
}