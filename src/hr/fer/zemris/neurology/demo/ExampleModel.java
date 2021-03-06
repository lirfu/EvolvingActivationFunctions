package hr.fer.zemris.neurology.demo;

import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import hr.fer.zemris.neurology.dl4j.IReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class ExampleModel implements IModel {
    private TrainParams params_;
    private MultiLayerNetwork model_;

    /**
     * Defines and builds the model.
     *
     * @param params      Parameters used in the training process.
     * @param layers      Array of sizes for the hidden layers.
     * @param activations Array of activation functions. Must define either one common activation activation function) or one activation function per layer.
     */
    public ExampleModel(@NotNull TrainParams params, @NotNull int[] layers, @NotNull IActivation[] activations) {
        boolean common_act = false;
        if (layers.length > 1 && activations.length == 1) { // Single common activation.
            common_act = true;
        } else if (layers.length > 0 && activations.length == 0 || activations.length != layers.length) {
            throw new IllegalArgumentException("Activation activation function ill defined! Please provide one common activation function or one activation function per layer.");
        }

        NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(params.seed()) // Reproducibility of results (defined initialization).
                .cacheMode(CacheMode.DEVICE)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(new StepSchedule(ScheduleType.EPOCH, params.learning_rate(), params.decay_rate(), params.decay_step())))
                .l2(params.regularization_coef())
                .dropOut(params.dropout_keep_prob());

        if (common_act) {
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
            if (!common_act) {
                lay.activation(activations[index]);
            }
            list.layer(index++, lay.build());
            last_size = l;
        }
        // Define output layer and loss.
        list.layer(index, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(last_size)
                .nOut(params.output_size())
                .build());

        model_ = new MultiLayerNetwork(list.build());
        params_ = params;
    }

    private ExampleModel(@NotNull ExampleModel model) {
        model_ = model.model_.clone();
        params_ = model.params_;
    }

    public void train(@NotNull DataSetIterator dataset, @NotNull ILogger log) {
        model_.init();

        final Stopwatch s = new Stopwatch();
        model_.setListeners(new BaseTrainingListener() { // Print the score at the start of each epoch.
            private int last_epoch_ = -1;

            @Override
            public void iterationDone(org.deeplearning4j.nn.api.Model model, int iteration, int epoch) {
                if (epoch != last_epoch_) {
                    last_epoch_ = epoch;
                    log.d("Epoch " + epoch + " has loss: " + model.score() + "   (" + Utilities.formatMiliseconds(s.lap()) + ")");
                }
            }
        });
        log.d("Training...");
        s.start();
        for (int i = 0; i < params_.epochs_num(); i++) {
            if (dataset.resetSupported()) {
                dataset.reset();
            }
            model_.fit(dataset);
        }
        log.d("Done! (" + Utilities.formatMiliseconds(s.stop()) + ")");
    }

    public void test(@NotNull DataSetIterator dataset, @NotNull ILogger log, @Nullable IReport report) {
        if (dataset.resetSupported()) {
            dataset.reset();
        }
        log.d("Evaluating...");
        Evaluation eval = new Evaluation(params_.output_size());
        ROCMultiClass roc = new ROCMultiClass(0);
        while (dataset.hasNext()) {
            DataSet next = dataset.next();
            INDArray output = model_.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
            roc.eval(next.getLabels(), output);
        }
        if (report != null) {
            report.build(params_.name(), this, eval, roc, dataset);
        }
    }

    /**
     * Predicts inputs from given dataset and logs as objects the input-output pairs - Pair(INDArray,INDArray).
     **/
    public void predict(@NotNull DataSetIterator dataset, @NotNull ILogger log) {
        if (dataset.resetSupported()) {
            dataset.reset();
        }
        while (dataset.hasNext()) {
            INDArray in = dataset.next().getFeatures();
            INDArray out = model_.output(in);
            log.o(new Pair<>(in, out));
        }
    }

    public void store(@NotNull String filepath) throws IOException {
        File locationToSave = new File(filepath);
        ModelSerializer.writeModel(model_, locationToSave, true);
    }

    public void load(@NotNull String filepath) throws IOException {
        model_ = ModelSerializer.restoreMultiLayerNetwork(filepath);
    }

    public ExampleModel clone() {
        return new ExampleModel(this);
    }

    @Override
    public void setModel(MultiLayerNetwork m) {
        model_ = m;
    }

    @Override
    public List<Double> getTrainLosses() {
        // TODO Auto generated block.
        return null;
    }

    @Override
    public List<Double> getTestLosses() {
        // TODO Auto generated block.
        return null;
    }
}
