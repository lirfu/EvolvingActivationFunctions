package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Triple;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.evaluation.classification.Evaluation;
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
     * @param train_path
     * @param test_path
     * @param params_builder
     */
    public TrainProcedureDL4J(@NotNull String train_path, @NotNull String test_path, @NotNull TrainParams.Builder params_builder) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        loadDatasets(train_path, test_path);

        params_ = params_builder
                .name(StorageManager.dsNameFromPath(train_path, false))
                .input_size(train_set_.numInputs())
                .output_size(train_set_.numOutcomes())
                .build();

        initialize();
    }

    public TrainProcedureDL4J(EvolvingActivationParams params) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.FLOAT);

        String train_path = params.train_path(), test_path = params.test_path();
        loadDatasets(train_path, test_path);

        // Automatically define necessary parameters.
        params.name(StorageManager.dsNameFromPath(train_path, false));
        params.input_size(train_set_.numInputs());
        params.output_size(train_set_.numOutcomes());
        params_ = params;

        initialize();
    }

    private void loadDatasets(String train_path, String test_path) throws IOException, InterruptedException {
        if (train_path.contains(";")) {  // Separate features and labels file.
            String[] files = train_path.split(";");
            train_set_ = StorageManager.loadSeparateCsvDataset(files);
        } else {  // Single file.
            train_set_ = train_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(train_path) : StorageManager.loadEntireCsvDataset(train_path);
        }

        if (test_path.contains(";")) {  // Separate features and labels file.
            String[] files = test_path.split(";");
            test_set_ = StorageManager.loadSeparateCsvDataset(files);
        } else {  // Single file.
            test_set_ = test_path.endsWith(".arff") ? StorageManager.loadEntireArffDataset(test_path) : StorageManager.loadEntireCsvDataset(test_path);
        }
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
        sb.append("  Classes (").append(labels.length).append("): ").append(Arrays.toString(labels)).append('\n');
        sb.append("Train set (").append(train_set_.numExamples()).append("): ").append(Arrays.toString(train_labels)).append('\n');
        if (val_labels != null)
            sb.append("Valid set (").append(validation_set_.numExamples()).append("): ").append(Arrays.toString(val_labels)).append('\n');
        sb.append(" Test set (").append(test_set_.numExamples()).append("): ").append(Arrays.toString(test_labels)).append('\n');
        sb.append("    Total (").append(total_instances).append("): ").append(Arrays.toString(total_labels)).append('\n');
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

    private void train_internal(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage, @NotNull DataSet train) {
        MultiLayerNetwork m = ((CommonModel) model).getModel();
        m.init();
        final Stopwatch timer = new Stopwatch();

        if (stats_storage != null) {  // Attach DL4J stats listener.
            m.addListeners(new StatsListener(stats_storage));
        }

        // Dataset copying in case of shuffling.
        Random random = new Random(42);
        DataSet set;
        if (params_.shuffle_batches()) {
            synchronized (train) {  // Shuffle dataset.
                set = train.copy();
            }
        } else {
            set = train; // No need for copying.
        }

        LinkedList<Double> train_losses = new LinkedList<>();
        LinkedList<Double> test_losses = new LinkedList<>();

        DataSetIterator iter = new TestDataSetIterator(set, params_.batch_size());
        timer.start();

        for (int i = 0; i < params_.epochs_num(); i++) {
            if (params_.shuffle_batches()) {
                set.shuffle(random.nextLong());
            }
            m.fit(iter);
            double train_loss = m.score(train_set_);
            double test_loss = m.score(validation_set_);
            train_losses.add(train_loss);
            test_losses.add(test_loss);
            log.d("Epoch " + (i + 1) + " has losses: " + train_loss + " - " + test_loss + "   (" + Utilities.formatMiliseconds(timer.lap()) + ")");

            if (!Double.isFinite(train_loss)) { // End training if network misbehaves.
                log.d("Training aborted! Model score became non-finite. " + train_loss);
                break;
            }
        }

        ((CommonModel) model).setTrainLosses(train_losses);
        ((CommonModel) model).setTestLosses(test_losses);
    }

    @Override
    public void train(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        train_internal(model, log, stats_storage, train_set_);
    }

    private DataSet construct_joined_dataset() {
        LinkedList<DataSet> dss = new LinkedList<>();
        dss.add(train_set_);
        if (validation_set_ != null)
            dss.add(validation_set_);
        return DataSet.merge(dss);
    }

    /**
     * Trains on joined train and test dataset.
     */
    @Override
    public void train_joined(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        DataSet joined_ds = construct_joined_dataset();
        train_internal(model, log, stats_storage, joined_ds);
    }

    /**
     * Trains the model and chooses the model and iteration where the test error was best.
     * Trains until max epochs done or convergence/overfitting detected.
     * Remembers the network with the best test loss and sets it to the given model.
     *
     * @return Optimal number of iterations.
     */
    public int train_itersearch(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage) {
        MultiLayerNetwork m = ((CommonModel) model).getModel();
        m.init();
        final Stopwatch timer = new Stopwatch();

        if (stats_storage != null) {  // Attach DL4J stats listener.
            m.addListeners(new StatsListener(stats_storage));
        }

        // Dataset copying in case of shuffling.
        Random random = new Random(42);
        DataSet set;
        if (params_.shuffle_batches()) {
            synchronized (train_set_) {
                set = train_set_.copy();
            }
        } else {
            set = train_set_; // No need for copying.
        }

        int impatience = 0;
        int best_epoch = 0;
        double best_train_loss = Double.MAX_VALUE;
        double best_test_loss = Double.MAX_VALUE;
        MultiLayerNetwork best_net = null;
        LinkedList<Double> train_losses = new LinkedList<>();
        LinkedList<Double> test_losses = new LinkedList<>();

        DataSetIterator train_iter = new TestDataSetIterator(set, params_.batch_size());
        timer.start();

        int i;
        for (i = 0; i < params_.epochs_num() && impatience < params_.train_patience(); i++) {
            if (params_.shuffle_batches()) {  // Shuffle dataset.
                set.shuffle(random.nextLong());
            }

            m.fit(train_iter);
            double train_loss = m.score(train_set_);
            double test_loss = m.score(validation_set_);
            train_losses.add(train_loss);
            test_losses.add(test_loss);
            log.d("Epoch " + (i + 1) + " has losses: " + train_loss + " - " + test_loss + "   (" + Utilities.formatMiliseconds(timer.lap()) + ")");

            if (!Double.isFinite(train_loss)) { // End training if network misbehaves.
                log.d("Training aborted! Model score became non-finite: " + train_loss);
                break;
            }

            boolean[] ts = new boolean[4];
            if (Math.abs((train_loss - best_train_loss) / best_train_loss) <= params_.convergence_delta()) {  // Convergence detection (relative difference).
                ts[0] = true;
                ts[1] = true;
            }
            if (train_loss > best_train_loss) {  // Divergence detection.
                ts[0] = true;
                ts[2] = true;
            }
            if (test_loss >= best_test_loss) {  // Overfitting detection.
                ts[0] = true;
                ts[3] = true;
            }
            if (ts[0]) {
                impatience++;
                log.d("Becoming impatient ("
                        + (ts[1] ? "cvg" : "___") + ","
                        + (ts[2] ? "dvg" : "___") + ","
                        + (ts[3] ? "overfit" : "_______")
                        + "): " + impatience);
            } else {
                impatience = 0;
            }

            if (train_loss < best_train_loss) {  // Update best train loss.
                best_train_loss = train_loss;
            }

            if (test_loss < best_test_loss) {  // Update best model over test loss.
                best_epoch = i + 1;
                best_test_loss = test_loss;
                best_net = m.clone();
            }
        }

        if (i >= params_.epochs_num()) {
            log.d("Training ended! Max iterations achieved!");
        }
        if (impatience >= params_.train_patience()) {
            log.d("Training aborted! Lost my patience!");
        }

        ((CommonModel) model).setTrainLosses(train_losses);
        ((CommonModel) model).setTestLosses(test_losses);

        if (best_net != null)  // Update with best model.
            model.setModel(best_net);

        return best_epoch;  // Return the optimal epoch number.
    }

    private Pair<ModelReport, Object> test_internal(@NotNull IModel model, DataSet dataset, int batch_size) {
        MultiLayerNetwork m = ((CommonModel) model).getModel();
        DataSetIterator it = new TestDataSetIterator(dataset, batch_size);

        Evaluation eval = new Evaluation(params_.output_size());
//        ROCMultiClass roc = new ROCMultiClass(0);

        m.doEvaluation(it, eval/*, roc*/);

        ModelReport report = new ModelReport();
        report.build(params_.name(), model, eval, null, it);

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

    private void earlystop_internal(@NotNull IModel model, @NotNull ILogger log, @Nullable StatsStorageRouter stats_storage, @NotNull DataSet train, @NotNull DataSet test) {
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

        timer.start();
        int patience = 0;
        double best_res = Double.MAX_VALUE;
        MultiLayerNetwork best_net = null;
        DataSetIterator iter = new TestDataSetIterator(train, params_.batch_size());

        for (int i = 0; i < params_.epochs_num() && patience < params_.train_patience() && running_good[0]; i++) {
            m.fit(iter);
            double res = m.score(test);

            if (res < best_res) {
                best_res = res;
                best_net = m.clone();

                if (Math.abs(m.score() - best_net.score()) <= params_.convergence_delta()) {  // Convergence detected.
                    patience++;
                    log.d("Patience (convergence): " + patience);
                } else {
                    patience = 0;
                }
            } else {  // Overfitting detected.
                patience++;
                log.d("Patience (overfitting): " + patience);
            }
        }

        if (!running_good[0]) {
            log.d("Training aborted! Model score became non-finite. " + m.score());
        }

        model.setModel(best_net);
    }


    private void collectModelActivations_internal(CommonModel model, Context c, Triple<Double, Double, Integer> range_n_buckets, DataSet ds) throws IOException {
        MultiLayerNetwork m = model.getModel();
        DataSetIterator iter = new TestDataSetIterator(ds, params_.batch_size());

        // Prepare layers for measuring.
        int layer_num = 0;
        for (Layer lay : m.getLayers()) {
            if (params_.batch_norm() && lay instanceof MyBatchNormalization
                    || !params_.batch_norm() && lay instanceof MyDenseLayer) {
                ((IOpenLayer) lay).setMeasuring(true);
                layer_num++;
            }
        }

        double min = range_n_buckets.getFirst(), max = range_n_buckets.getSecond();
        int bucket_num = range_n_buckets.getThird();
        long[][] layerwise_buckets = new long[layer_num][bucket_num];

        // Line for bucket sorting.
        double k = (bucket_num - 2) / (max - min);
        double l = bucket_num / 2. - k * (max + min) / 2.;

        // Collect activations.
        while (iter.hasNext()) {
            // Forward propagate to remember activations.
            DataSet batch = iter.next();
            m.output(batch.getFeatures());

            // Layerwise count bucket occasions.
            int l_i = 0;
            for (Layer lay : m.getLayers()) {
                if (params_.batch_norm() && lay instanceof MyBatchNormalization
                        || !params_.batch_norm() && lay instanceof MyDenseLayer) {
                    INDArray act = ((IOpenLayer) lay).getActivation();

                    for (int i = 0; i < act.length(); i++) {
                        // Calculate index from defined line.
                        int b_i = Math.max(0, Math.min(bucket_num - 1, (int) Math.floor(l + k * act.getDouble(i))));
                        layerwise_buckets[l_i][b_i]++;
                    }
                    l_i++;
                }
            }
        }

        // Store buckets.
        StorageManager.storeActivations(layerwise_buckets, range_n_buckets, c);

        // Close layers for measuring.
        for (Layer lay : m.getLayers()) {
            if (params_.batch_norm() && lay instanceof MyBatchNormalization
                    || !params_.batch_norm() && lay instanceof MyDenseLayer) {
                ((IOpenLayer) lay).setMeasuring(false);
            }
        }
    }

    public void collectModelActivationsOnTrain(CommonModel model, Context c, Triple<Double, Double, Integer> range_n_buckets) throws IOException {
        collectModelActivations_internal(model, c, range_n_buckets, train_set_);
    }

    public void collectModelActivationsOnTrainJoined(CommonModel model, Context c, Triple<Double, Double, Integer> range_n_buckets) throws IOException {
        collectModelActivations_internal(model, c, range_n_buckets, construct_joined_dataset());
    }

    public void collectModelActivationsOnTest(CommonModel model, Context c, Triple<Double, Double, Integer> range_n_buckets) throws IOException {
        collectModelActivations_internal(model, c, range_n_buckets, test_set_);
    }

    public void release_train() {
        train_set_ = null;
    }

    @Override
    public void storeResults(IModel model, Context context, Pair<ModelReport, Object> result) throws IOException {
//        StorageManager.storeModel(((CommonModel) model), context);
        StorageManager.storeTrainParameters(params_, context);
        StorageManager.storeResults(result.getKey(), context);
//        StorageManager.storePredictions((INDArray) result.getSecond(), context);
    }

    /**
     * Runs a UIServer and attaches the given stats storage.
     */
    public static void displayTrainStats(FileStatsStorage storage) {
        UIServer uiServer_ = UIServer.getInstance();
        uiServer_.attach(storage);
    }
}
