package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;

/**
 * A common procedure for learning on a particular dataset.
 */
public class TrainProcedureDL4J implements ITrainProcedure {
    private DataSet train_set_, validation_set_ = null, test_set_;
    private TrainParams params_;

    /**
     * Use given splitted dataset to train and test the network.
     *
     * @param train_set_path
     * @param test_set_path
     * @param params_builder
     */
    public TrainProcedureDL4J(@NotNull String train_set_path, @NotNull String test_set_path, @NotNull TrainParams.Builder params_builder) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        train_set_ = train_set_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(train_set_path) : StorageManager.loadEntireCsvDataset(train_set_path);
        test_set_ = test_set_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(test_set_path) : StorageManager.loadEntireCsvDataset(test_set_path);

        params_ = params_builder
                .name(StorageManager.dsNameFromPath(train_set_path, false))
                .input_size(train_set_.numInputs())
                .output_size(train_set_.numOutcomes())
                .build();

        initialize();
    }

    public TrainProcedureDL4J(EvolvingActivationParams params) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        String train_path = params.train_path(), test_path = params.test_path();
        train_set_ = train_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(train_path) : StorageManager.loadEntireCsvDataset(train_path);
        test_set_ = test_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(test_path) : StorageManager.loadEntireCsvDataset(test_path);

        // Automatically define necessary parameters.
        params.name(StorageManager.dsNameFromPath(train_path, false));
        params.input_size(train_set_.numInputs());
        params.output_size(train_set_.numOutcomes());
        params_ = params;

        initialize();
    }

    /**
     * Splits train dataset if necessary and applies feature normalization if necessary.
     */
    private void initialize() {
        boolean split = params_.train_percentage() > 0 && params_.train_percentage() < 1;

        // Normalize both sets on entire train before splitting.
        if (params_.normalize_features()) {
            NormalizerStandardize norm = new NormalizerStandardize();
            norm.fit(train_set_);
            norm.transform(train_set_);
            norm.transform(test_set_);
        }

        if (split) { // Split train set into train and validation
            train_set_.shuffle(42);
            SplitTestAndTrain splitter = train_set_.splitTestAndTrain(params_.train_percentage());
            train_set_ = splitter.getTrain();
            validation_set_ = splitter.getTest();
        }
    }

    /**
     * Limit periodicity of calls for garbage collector.
     *
     * @param period Period of GC calls (in milliseconds). Using 0 or negative disables this option.
     */
    public TrainProcedureDL4J callGCPeriod(int period) {
        if (period > 0) {
            Nd4j.getMemoryManager().setAutoGcWindow(period);
            Nd4j.getMemoryManager().togglePeriodicGc(true);
        } else {
            // Assertion error in DL4J.
            //  Nd4j.getMemoryManager().setAutoGcWindow(0);
            Nd4j.getMemoryManager().togglePeriodicGc(false);
        }
        return this;
    }

    @Override
    public String describeDatasets() {
        int[] labels = new int[train_set_.numOutcomes()];
        for (int i = 0; i < labels.length; i++)
            labels[i] = i;

        int total_instances = 0;
        int[] total_labels = new int[train_set_.numOutcomes()];

        int[] train_labels = new int[train_set_.numOutcomes()];
        INDArray arr = train_set_.getLabels().argMax(1);
        for (int i = 0; i < train_set_.numExamples(); i++) {
            int l = arr.getInt(i);
            train_labels[l]++;
            total_labels[l]++;
            total_instances++;
        }

        int[] test_labels = new int[test_set_.numOutcomes()];
        arr = test_set_.getLabels().argMax(1);
        for (int i = 0; i < test_set_.numExamples(); i++) {
            int l = arr.getInt(i);
            test_labels[l]++;
            total_labels[l]++;
            total_instances++;
        }

        int[] val_labels = null;
        if (validation_set_ != null) {
            val_labels = new int[validation_set_.numOutcomes()];
            arr = validation_set_.getLabels().argMax(1);
            for (int i = 0; i < validation_set_.numExamples(); i++) {
                int l = arr.getInt(i);
                val_labels[l]++;
                total_labels[l]++;
                total_instances++;
            }
        }


        StringBuilder sb = new StringBuilder();
        sb.append("  Classes: ").append(Arrays.toString(labels)).append('\n');
        sb.append("Train set: ").append(Arrays.toString(train_labels)).append('\n');
        if (val_labels != null)
            sb.append("Valid set: ").append(Arrays.toString(val_labels)).append('\n');
        sb.append(" Test set: ").append(Arrays.toString(test_labels)).append('\n');
        sb.append("    Total: ").append(Arrays.toString(total_labels)).append('\n');
        sb.append("Total inp: ").append(total_instances).append('\n');
        return sb.toString();
    }

    @Override
    public Context createContext(String experiment_name) {
        return new Context(params_.name(), experiment_name);
    }

    /**
     * Defines and builds the model with internal common parameters.
     *
     * @param architecture Network architecture.
     * @param activations  Array of activation functions. Must define either one function per layer or a single common activation function.
     */
    @Override
    public CommonModel createModel(NetworkArchitecture architecture, IActivation[] activations) {
        return new CommonModel(params_, architecture, activations);
    }

    private void train_internal(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage, @NotNull DataSet dataset) {
        MultiLayerNetwork m = ((CommonModel) model).getModel();
        m.init();
        final Stopwatch timer = new Stopwatch();
        final boolean[] running_good = {true};

        if (stats_storage != null) {
            m.addListeners(new StatsListener(stats_storage));
        }
        m.addListeners(new BaseTrainingListener() { // Print the score at the start of each epoch.
            private int last_epoch_ = -1;

            @Override
            public void iterationDone(org.deeplearning4j.nn.api.Model model, int iteration, int epoch) {
                if (!Double.isFinite(model.score())) { // End training if network misbehaves.
                    running_good[0] = false;
                }
                if (epoch != last_epoch_) {
                    last_epoch_ = epoch;
                    log.d("Epoch " + (epoch + 1) + " has loss: " + model.score() + "   (" + Utilities.formatMiliseconds(timer.lap()) + ")");
                }
            }
        });

        Random random = new Random(42);
        DataSet set;
//        if (params_.shuffle_batches()) {
        synchronized (dataset) {
            set = dataset.copy();
        }
//        } else {
//            set = dataset; // No need for copying.
//        }

        timer.start();
        DataSetIterator iter = new TestDataSetIterator(set, params_.batch_size());
        for (int i = 0; i < params_.epochs_num() && running_good[0]; i++) {
            if (params_.shuffle_batches()) {
                set.shuffle(random.nextLong());
            }
            m.fit(iter);
        }
    }

    @Override
    public void train(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        train_internal(model, log, stats_storage, train_set_);
    }

    /**
     * Trains on joined train and test dataset.
     */
    @Override
    public void train_joined(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        LinkedList<DataSet> dss = new LinkedList<>();
        dss.add(train_set_);
        if (validation_set_ != null)
            dss.add(validation_set_);
        train_internal(model, log, stats_storage, DataSet.merge(dss));
    }

    private Pair<ModelReport, Object> test_internal(@NotNull IModel model, DataSet dataset, int batch_size) {
        MultiLayerNetwork m = ((CommonModel) model).getModel();
        DataSetIterator it = new TestDataSetIterator(dataset, batch_size);

        Evaluation eval = new Evaluation(params_.output_size());
        ROCMultiClass roc = new ROCMultiClass(0);

        m.doEvaluation(it, eval, roc);

        ModelReport report = new ModelReport();
        report.build(params_, m, eval, roc);

        return new Pair<>(report, m.output(dataset.getFeatures()));
    }

    @Override
    /** Validates current model on validation set. If validation set isn't defined, returns result of method <code>test()</code>. */
    public Pair<ModelReport, Object> validate(@NotNull IModel model) {
        if (validation_set_ != null)
            return test_internal(model, validation_set_, params_.batch_size());
        return test(model);
    }

    @Override
    public Pair<ModelReport, Object> test(@NotNull IModel model) {
        return test_internal(model, test_set_, params_.batch_size());
    }

    public Pair<ModelReport, Object> test(@NotNull IModel model, int batch_size) {
        return test_internal(model, test_set_, batch_size);
    }

    @Override
    public void storeResults(IModel model, Context context, Pair<ModelReport, Object> result) throws IOException {
//        StorageManager.storeModel(((CommonModel) model), context);
        StorageManager.storeTrainParameters(params_, context);
        StorageManager.storeResults(result.getKey(), context);
//        StorageManager.storePredictions((INDArray) result.getVal(), context);
    }

    /**
     * Runs a UIServer and attaches the given stats storage.
     */
    public static void displayTrainStats(FileStatsStorage storage) {
        UIServer uiServer_ = UIServer.getInstance();
        uiServer_.attach(storage);
    }
}
