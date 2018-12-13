package hr.fer.zemris.evolveactivationfunction.architecturesearch;

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
        String train_ds = "res/noiseless/5k/9class/noiseless_9class_5k_train.arff";
        String test_ds = "res/noiseless/5k/9class/noiseless_9class_5k_test.arff";

        IActivation common_activation = new ActivationReLU();

        TrainParams.Builder params = new TrainParams.Builder()
                .batch_size(32)
                .normalize_features(true)
                .shuffle_batches(true)
                .epochs_num(80)
                .learning_rate(2e-3)
                .decay_rate(1 - 1e-2)
                .decay_step(5)
//                .regularization_coef(1e-4)
                .dropout_keep_prob(0.5)
                ;

        TrainProcedure train_procedure = new TrainProcedure(train_ds, test_ds, params);
        CommonModel model = train_procedure.createModel(new int[]{30, 30}, new IActivation[]{common_activation});
        Context context = train_procedure.createContext("7_relu_30_30_drop");

        ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
        Stopwatch timer = new Stopwatch();

        timer.start();
        train_procedure.train(model, log, stat_storage);
        Pair<ModelReport, INDArray> result = train_procedure.test(model);
        log.logD(result.getKey().toString());
        log.logD("Elapsed time: " + Utilities.formatMiliseconds(timer.stop()));

        train_procedure.storeResults(model, context, result);

        TrainProcedure.displayTrainStats(stat_storage);
    }
}
