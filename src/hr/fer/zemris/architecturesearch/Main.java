package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.CommonModel;
import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.TrainProcedure;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public class Main {
    // Modifying weights (net2net): https://stackoverflow.com/questions/42806761/initialize-custom-weights-in-deeplearning4j

    public static void main(String[] args) throws IOException, InterruptedException {
        String train_ds = "res/noiseless/100k/9class/noiseless_9class_100k_train.arff";
        String test_ds = "res/noiseless/100k/9class/noiseless_9class_100k_test.arff";

        IActivation common_activation = new ActivationReLU();

        for (Experiment e : experiments) {
            TrainProcedure train_procedure = new TrainProcedure(train_ds, test_ds, e.getParams());
            CommonModel model = train_procedure.createModel(e.getArchitecture(), new IActivation[]{common_activation});
            Context context = train_procedure.createContext(e.getName());

            ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
            FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
            Stopwatch timer = new Stopwatch();

            timer.start();
            log.logD("Training...");
            train_procedure.train(model, log, stat_storage);
            log.logD("Testing...");
            Pair<ModelReport, INDArray> result = train_procedure.test(model);
            log.logD(result.getKey().toString());
            log.logD("Elapsed time: " + Utilities.formatMiliseconds(timer.stop()));

            train_procedure.storeResults(model, context, result);
        }

//        TrainProcedure.displayTrainStats(stat_storage);
    }

    private static final Experiment[] experiments = {
            new Experiment("01_relu_30_30_overfit", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .epochs_num(100)
                    .learning_rate(1e-3)),
            new Experiment("02_relu_30_30_overfit_normfeat", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .normalize_features(true)
                    .epochs_num(100)
                    .learning_rate(1e-3)),
            new Experiment("03_relu_30_30_overfit_normfeat_shflb", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .normalize_features(true)
                    .shuffle_batches(true)
                    .epochs_num(100)
                    .learning_rate(1e-3)),
            new Experiment("04_relu_30_30_l2_long", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .normalize_features(true)
                    .shuffle_batches(true)
                    .epochs_num(100)
                    .learning_rate(1e-3)
                    .regularization_coef(1e-4)),
            new Experiment("05_relu_30_30_l2", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .normalize_features(true)
                    .shuffle_batches(true)
                    .epochs_num(80)
                    .learning_rate(1e-3)
                    .regularization_coef(1e-4)),
            new Experiment("06_relu_30_30_l2_decay", new int[]{30, 30}, new TrainParams.Builder()
                    .batch_size(32)
                    .normalize_features(true)
                    .shuffle_batches(true)
                    .epochs_num(80)
                    .learning_rate(2e-3)
                    .regularization_coef(1e-4)
                    .decay_rate(1 - 1e-2)
                    .decay_step(5))
    };
}
