package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.DataSetUtils;

import static org.junit.Assert.*;

public class ModelReportTest {

    @Test
    public void testGainEntropies() {
        // Build a Hamming weight counting dataset.
        INDArray f = Nd4j.zeros(4, 2);
        INDArray l = Nd4j.zeros(4, 4);

        f.putScalar(new int[]{0, 0}, 0);
        f.putScalar(new int[]{0, 1}, 0);

        l.putScalar(new int[]{0, 0}, 1);
        l.putScalar(new int[]{0, 1}, 0);
        l.putScalar(new int[]{0, 2}, 0);
        l.putScalar(new int[]{0, 3}, 0);

        f.putScalar(new int[]{1, 0}, 0);
        f.putScalar(new int[]{1, 1}, 1);

        l.putScalar(new int[]{1, 0}, 0);
        l.putScalar(new int[]{1, 1}, 1);
        l.putScalar(new int[]{1, 2}, 0);
        l.putScalar(new int[]{1, 3}, 0);

        f.putScalar(new int[]{2, 0}, 1);
        f.putScalar(new int[]{2, 1}, 0);

        l.putScalar(new int[]{2, 0}, 0);
        l.putScalar(new int[]{2, 1}, 0);
        l.putScalar(new int[]{2, 2}, 1);
        l.putScalar(new int[]{2, 3}, 0);

        f.putScalar(new int[]{3, 0}, 1);
        f.putScalar(new int[]{3, 1}, 1);

        l.putScalar(new int[]{3, 0}, 0);
        l.putScalar(new int[]{3, 1}, 0);
        l.putScalar(new int[]{3, 2}, 0);
        l.putScalar(new int[]{3, 3}, 1);

        DataSet ds = new DataSet(f, l);
        DataSetIterator it = new TestDataSetIterator(ds, 1);

        System.out.println("Features:");
        System.out.println(f);
        System.out.println("Labels:");
        System.out.println(l);

        // Define network.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.3))
                .seed(42)
                .biasInit(1)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(4)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(it, 10);

        // Network info.
//        System.out.println("=====> Network info:");
//        System.out.println(net.summary());

        // Predictions.
        System.out.println("=====> Predictions:");
        it = new TestDataSetIterator(ds, 2);
        it.reset();
        INDArray pred = net.output(it);
        System.out.println(pred);

        // Results.
        Evaluation eval = new Evaluation((int) l.size(1));
        net.doEvaluation(it, eval);

        ModelReport rep = new ModelReport();
        rep.build("name", new CommonModel(net), eval, null, it);

        System.out.println("\n=====> Report:");
        System.out.println(rep);
    }
}