package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.threading.Worker;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.IOException;
import java.util.Random;

/**
 * A common procedure for learning on a particular dataset.
 */
public class TrainProcedure {
    private static final long SEED = 42;

    private DataSet train_set_, test_set_;
    private TrainParams params_;

    public TrainProcedure(@NotNull String train_set_path, @NotNull String test_set_path, @NotNull TrainParams.Builder params_builder) throws IOException, InterruptedException {
        train_set_ = StorageManager.loadEntireDataset(train_set_path);
        test_set_ = StorageManager.loadEntireDataset(test_set_path);

        params_ = params_builder
                .seed(SEED)
                .name(StorageManager.dsNameFromPath(train_set_path))
                .input_size(train_set_.numInputs())
                .output_size(train_set_.numOutcomes())
                .build();

        // Normalize dataset.
        if (params_.normalize_features()) {
            DataNormalization norm_ = new NormalizerStandardize();
            norm_.fit(train_set_);
            norm_.transform(train_set_);
            norm_.transform(test_set_);
        }
    }

    /**
     * Constructs an iterator from the given dataset instance.
     *
     * @param dataset    The dataset to construct batches from.
     * @param batch_size The batch size.
     * @return Iterator that splits the dataset into batches.
     */
    public static DataSetIterator batch(DataSet dataset, int batch_size) {
        return new ListDataSetIterator<>(dataset.asList(), batch_size);
    }

    public Context createContext(String experiment_name) {
        return new Context(params_.name(), experiment_name);
    }

    /**
     * Defines and builds the model with internal common parameters.
     *
     * @param architecture Array of sizes for each hidden layer.
     * @param activations  Array of activation functions. Must define either one function per layer or a single common activation function.
     */
    public CommonModel createModel(int[] architecture, IActivation[] activations) {
        return new CommonModel(params_, architecture, activations);
    }

    public void train(@NotNull CommonModel model, @NotNull ILogger log, @NotNull StatsStorageRouter stats_storage) {
        log.logD("Training...");
        MultiLayerNetwork m = model.getModel();
        m.init();

        m.setListeners(new StatsListener(stats_storage), new BaseTrainingListener() { // Print the score at the start of each epoch.
            private int last_epoch_ = -1;

            @Override
            public void iterationDone(org.deeplearning4j.nn.api.Model model, int iteration, int epoch) {
                if (epoch != last_epoch_) {
                    last_epoch_ = epoch;
                    log.logD("Epoch " + (epoch + 1) + " has loss: " + model.score());
                }
            }
        });

        Random random = new Random(params_.seed());
        DataSet set;
        if (params_.shuffle_batches()) {
            set = train_set_.copy();
        } else {
            set = train_set_; // No need for copying.
        }

        for (int i = 0; i < params_.epochs_num(); i++) {
            if (params_.shuffle_batches()) {
                set.shuffle(random.nextLong());
            }
            DataSetIterator batches = batch(set, params_.batch_size());
            m.fit(batches);
        }
    }

    public Pair<ModelReport, INDArray> test(@NotNull CommonModel model) {
        MultiLayerNetwork m = model.getModel();
        INDArray output = m.output(test_set_.getFeatures());

        Evaluation eval = new Evaluation(params_.output_size());
        eval.eval(test_set_.getLabels(), output);

        ROCMultiClass roc = new ROCMultiClass(0);
        roc.eval(test_set_.getLabels(), output);

        ModelReport report = new ModelReport();
        report.build(params_, m, eval, roc);

        return new Pair<>(report, output);
    }

    public void storeResults(CommonModel model, Context context, Pair<ModelReport, INDArray> result) throws IOException {
        StorageManager.storeModel(model, context);
        StorageManager.storeTrainParameters(params_, context);
        StorageManager.storeResults(result.getKey(), context);
        StorageManager.storePredictions(result.getVal(), context);
    }

    /**
     * Runs a UIServer and attaches the given stats storage.
     */
    public static void displayTrainStats(FileStatsStorage storage) {
        UIServer uiServer_ = UIServer.getInstance();
        uiServer_.attach(storage);
    }
}
