package hr.fer.zemris.neurology.dl4j;

import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CommonModel implements IModel {
    private Params params_;
    private MultiLayerNetwork model_;

    /**
     * Defines and builds the model.
     *
     * @param params      Parameters used in the training process.
     * @param layers      Array of sizes for the hidden layers.
     * @param activations Array of activation functions. Must define either one common activation function) or one function per layer.
     */
    public CommonModel(@NotNull Params params, @NotNull int[] layers, @NotNull IActivation[] activations) {
        boolean common_func = false;
        if (layers.length > 1 && activations.length == 1) { // Single common activation.
            common_func = true;
        } else if (layers.length > 0 && activations.length == 0 || activations.length != layers.length) {
            throw new IllegalArgumentException("Activation function ill defined! Please provide one common function or one function per layer.");
        }

        NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(params.seed()) // Reproducibility of results (defined initialization).
                .cacheMode(CacheMode.DEVICE)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(params.learning_rate()))
                .l2(params.regularization_coef()); // L2 parameter regularization.

        if (common_func) {
            conf.activation(activations[0]);
        }

        // Define inner layers.
        NeuralNetConfiguration.ListBuilder list = conf.list();
        int index = 0;
        int last_size = params.input_size();
        for (int l : layers) {
            DenseLayer.Builder lay = new DenseLayer.Builder()
                    .nIn(last_size)
                    .nOut(l);
            if (!common_func) {
                lay.activation(activations[index]);
            }
            list.layer(index++, lay.build());
            last_size = l;
        }
        // Define output layer and loss.
        list.layer(index, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                .activation(Activation.SOFTMAX)
                .nIn(last_size)
                .nOut(params.output_size())
                .build());

        model_ = new MultiLayerNetwork(list.build());
        params_ = params;
    }

    @Override
    public void train(@NotNull DataSetIterator dataset, @NotNull ILogger log) {
        model_.init();
        model_.setListeners(new BaseTrainingListener() { // Print the score with start of every epoch.
            private int last_epoch_ = -1; // Because there's no other way of getting the dataset size -.-"

            @Override
            public void iterationDone(org.deeplearning4j.nn.api.Model model, int iteration, int epoch) {
                if (epoch != last_epoch_) {
                    last_epoch_ = epoch;
                    log.logD("Epoch " + epoch + " has loss: " + model.score());
                }
            }
        });
        log.logD("Training...");
        for (int i = 0; i < params_.epochs_num(); i++) {
            if (dataset.resetSupported()) {
                dataset.reset();
            }
            model_.fit(dataset);
        }
    }

    @Override
    public void test(@NotNull DataSetIterator dataset, @NotNull ILogger log) {
        if (dataset.resetSupported()) {
            dataset.reset();
        }
        log.logD("Evaluating...");
        log.logD("Model train loss: " + model_.score());
        Evaluation eval = new Evaluation(params_.output_size());
        while (dataset.hasNext()) {
            DataSet next = dataset.next();
            INDArray output = model_.output(next.getFeatures()); //next the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        log.logD(eval.stats());
    }

    /**
     * Predicts inputs from given dataset and logs as objects the input-output pairs - Pair(INDArray,INDArray).
     **/
    @Override
    public void predict(@NotNull DataSetIterator dataset, @NotNull ILogger log) {
        if (dataset.resetSupported()) {
            dataset.reset();
        }
        while (dataset.hasNext()) {
            INDArray in = dataset.next().getFeatures();
            INDArray out = model_.output(in);
            log.logO(new Pair<>(in, out));
        }
    }

    public static class Params {
        private final int input_size_;
        private final int output_size_;
        private int epochs_num_;
        private int batch_size_;
        private double learning_rate_;
        private double regularization_coef_;
        private long seed_;

        public Params(int input_size, int output_size, int epochs_num, int batch_size, double learning_rate, double regularization_coef, long seed) {
            input_size_ = input_size;
            output_size_ = output_size;
            epochs_num_ = epochs_num;
            batch_size_ = batch_size;
            learning_rate_ = learning_rate;
            regularization_coef_ = regularization_coef;
            seed_ = seed;
        }


        public int input_size() {
            return input_size_;
        }

        public int output_size() {
            return output_size_;
        }

        public int epochs_num() {
            return epochs_num_;
        }

        public int batch_size() {
            return batch_size_;
        }

        public double learning_rate() {
            return learning_rate_;
        }

        public double regularization_coef() {
            return regularization_coef_;
        }

        public long seed() {
            return seed_;
        }
    }
}
