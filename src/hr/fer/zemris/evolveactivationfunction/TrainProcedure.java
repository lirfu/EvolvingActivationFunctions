package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
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

    /**
     * Use given splitted dataset to train and test the network.
     *
     * @param train_set_path
     * @param test_set_path
     * @param params_builder
     */
    public TrainProcedure(@NotNull String train_set_path, @NotNull String test_set_path, @NotNull TrainParams.Builder params_builder) throws IOException, InterruptedException {
        train_set_ = train_set_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(train_set_path) : StorageManager.loadEntireCsvDataset(train_set_path);
        test_set_ = test_set_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(test_set_path) : StorageManager.loadEntireCsvDataset(test_set_path);
        initialize(StorageManager.dsNameFromPath(train_set_path), params_builder);
    }

    /**
     * Use given dataset and create a train-test split using given ratio.
     *
     * @param dataset_name
     * @param params_builder
     */
    public TrainProcedure(@NotNull String dataset_name, @NotNull TrainParams.Builder params_builder, float train_percentage) throws IOException, InterruptedException {
        DataSet ds = dataset_name.endsWith(".arff") ? StorageManager.loadEntireArffDataset(dataset_name) : StorageManager.loadEntireCsvDataset(dataset_name);
        ds.shuffle(42);
        SplitTestAndTrain split = ds.splitTestAndTrain(train_percentage);
        train_set_ = split.getTrain();
        test_set_ = split.getTest();

        params_builder.train_percentage(train_percentage);
        initialize(StorageManager.dsNameFromPath(dataset_name), params_builder);
    }

    private void initialize(String dataset_name, TrainParams.Builder params_builder) {
        params_ = params_builder
                .seed(SEED)
                .name(dataset_name)
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

    public void train(@NotNull CommonModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        MultiLayerNetwork m = model.getModel();
        m.init();
        final Stopwatch timer = new Stopwatch();

        if (stats_storage != null) {
            m.addListeners(new StatsListener(stats_storage));
        }
        m.addListeners(new BaseTrainingListener() { // Print the score at the start of each epoch.
            private int last_epoch_ = -1;

            @Override
            public void iterationDone(org.deeplearning4j.nn.api.Model model, int iteration, int epoch) {
                if (epoch != last_epoch_) {
                    last_epoch_ = epoch;
                    log.d("Epoch " + (epoch + 1) + " has loss: " + model.score() + "   (" + Utilities.formatMiliseconds(timer.lap()) + ")");
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

        timer.start();
        DataSetIterator iter = new TestDataSetIterator(set, params_.batch_size());
        for (int i = 0; i < params_.epochs_num(); i++) {
            if (params_.shuffle_batches()) {
                set.shuffle(random.nextLong());
            }
            m.fit(iter);
        }
    }

    public Pair<ModelReport, INDArray> test(@NotNull CommonModel model) {
        MultiLayerNetwork m = model.getModel();
        DataSetIterator it = new TestDataSetIterator(test_set_, params_.batch_size());

        Evaluation eval = new Evaluation(params_.output_size());
        ROCMultiClass roc = new ROCMultiClass(0);

        m.doEvaluation(it, eval, roc);

        ModelReport report = new ModelReport();
        report.build(params_, m, eval, roc);

        return new Pair<>(report, m.output(test_set_.getFeatures()));
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
