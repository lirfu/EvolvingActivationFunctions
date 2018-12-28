package hr.fer.zemris.neurology.demo;

import hr.fer.zemris.data.datasets.BinaryDecoderClassification;
import hr.fer.zemris.neurology.dl4j.*;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
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
            // calculate the activationfunction gradient in given input
            INDArray out = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(input));
            // multiply with incoming gradient (dL/do)
            out.muli(gradient);
            // gradients toward inputs and the act. func. inner parameters (or null if func. has no learnable params)
            return new Pair<>(out, null);
        }
    }

    public static void originalTest() throws IOException {
        TrainParams p = new TrainParams.Builder()
                .input_size(28 * 28).output_size(10).epochs_num(20).batch_size(64)
                .learning_rate(0.001).regularization_coef(1e-6).dropout_keep_prob(0.5)
                .decay_rate(9e-1).decay_step(5)
                .seed(42).build();

        DataSetIterator mnistTrain = new MnistDataSetIterator(p.batch_size(), true, (int) p.seed());
        DataSetIterator mnistTest = new MnistDataSetIterator(p.batch_size(), false, (int) p.seed());

        ExampleModel model = new ExampleModel(p, new int[]{500, 100}, new IActivation[]{new MyActivation()});
        StdoutLogger log = new StdoutLogger();

        model.train(mnistTrain, log);

        IReport rep = new ModelReport();
        model.test(mnistTest, log, rep);
        log.d(rep.toString());
    }

    public static void dataFromGeneratorTest() throws FileNotFoundException {
        DataSetIterator dataset = new Dl4jDataset(new BinaryDecoderClassification(), 3, 42);
        TrainParams p = new TrainParams.Builder()
                .input_size(dataset.inputColumns()).output_size(dataset.totalOutcomes())
                .epochs_num(500).batch_size(dataset.batch())
                .learning_rate(1e-1).regularization_coef(1e-4).dropout_keep_prob(0.5)
                .seed(42).build();

        ExampleModel model = new ExampleModel(p, new int[]{5, 5}, new IActivation[]{new MyActivation()});
        StdoutLogger log = new StdoutLogger();

        dataset.reset();
        model.train(dataset, log);
        dataset.reset();

        IReport rep = new ModelReport();
        model.test(dataset, log, rep);
        log.d(rep.toString());

        // Test storeResults/load mechanism.
        try {
            String path = "./model.zip";

            log.d("Storing model...");
            model.store(path);
            model = new ExampleModel(p, new int[]{1, 1, 1}, new IActivation[]{new ActivationReLU()});
            log.d("Loading model...");
            model.load(path);

            rep = new ModelReport();
            model.test(dataset, log, rep);
            log.d("Report of loaded model:");
            log.d(rep.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}