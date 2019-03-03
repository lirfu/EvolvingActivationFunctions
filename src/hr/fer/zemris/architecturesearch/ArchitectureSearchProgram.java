package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.CommonModel;
import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.TrainProcedure;
import hr.fer.zemris.evolveactivationfunction.activationfunction.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.nodes.AddNode;
import hr.fer.zemris.evolveactivationfunction.nodes.ConstNode;
import hr.fer.zemris.evolveactivationfunction.nodes.CustomReLUNode;
import hr.fer.zemris.evolveactivationfunction.nodes.InputNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Counter;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import hr.fer.zemris.utils.threading.WorkArbiter;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Random;

public class ArchitectureSearchProgram {
    // Modifying weights (net2net): https://stackoverflow.com/questions/42806761/initialize-custom-weights-in-deeplearning4j

    public static void main(String[] args) throws IOException, InterruptedException {
        String train_ds = "res/noiseless/5k/9class/noiseless_9class_5k_train.arff";
        String test_ds = "res/noiseless/5k/9class/noiseless_9class_5k_test.arff";

        IActivation common_activation;
        common_activation = new ActivationReLU();

//        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
//                .setNodeSet(new TreeNodeSet(new Random()))
////                .add(new ReLUNode())
//                .add(new CustomReLUNode())
//                .add(new InputNode())
//                .build();
//        common_activation = new CustomFunction(tree);

        final WorkArbiter arbiter = new WorkArbiter("Experimenter", 3);

        for (Experiment e : experiments) {
            arbiter.postWork(() -> {
                try {
                    TrainProcedure train_procedure = new TrainProcedure(train_ds, test_ds, e.getParams());
                    CommonModel model = train_procedure.createModel(e.getArchitecture(), new IActivation[]{common_activation});
                    Context context = train_procedure.createContext(e.getName());

                    ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
                    FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
                    Stopwatch timer = new Stopwatch();

                    timer.start();
                    log.d("Training...");
                    train_procedure.train(model, log, stat_storage);
                    log.d("Testing...");
                    Pair<ModelReport, INDArray> result = train_procedure.test(model);
                    log.d(result.getKey().toString());
                    log.d("Elapsed time: " + Utilities.formatMiliseconds(timer.stop()));
                    train_procedure.storeResults(model, context, result);
//                    train_procedure.displayTrainStats(stat_storage);
                } catch (IOException | InterruptedException e1) {
                    e1.printStackTrace();
                }
            });
        }

        System.out.println("Pending:\n" + arbiter.getStatus());

        arbiter.waitOn(arbiter.getFinishedCondition());
    }

    private static final Experiment[] experiments = {
            new Experiment("01_relu_30_30_changedmodel", "fc(30)-fc(30)", new TrainParams.Builder()
                    .batch_size(32)
                    .epochs_num(2)
                    .learning_rate(1e-3))
//            /* 30-30 */
//            new Experiment("01_relu_30_30_overfit", new int[]{30, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("02_relu_30_30_overfit_normfeat", new int[]{30, 30}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)),
//            new Experiment("03_relu_30_30_overfit_normfeat_shflb", new int[]{30, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("04_relu_30_30_l2_long", new int[]{30, 30}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .shuffle_batches(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)
////                    .regularization_coef(1e-4)),
//            new Experiment("05_relu_30_30_l2", new int[]{30, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)),
//            new Experiment("06_relu_30_30_l2_decay", new int[]{30, 30}, new TrainParams.Builder()
//                    .batch_size(64)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)
//                    .decay_rate(1 - 1e-2)
//                    .decay_step(1)),
//
//            /* 50-50 */
//            new Experiment("07_relu_50_50_overfit", new int[]{50, 50}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("08_relu_50_50_overfit_normfeat", new int[]{50, 50}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)),
//            new Experiment("09_relu_50_50_overfit_normfeat_shflb", new int[]{50, 50}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("10_relu_50_50_l2_long", new int[]{50, 50}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .shuffle_batches(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)
////                    .regularization_coef(1e-4)),
//            new Experiment("11_relu_50_50_l2", new int[]{50, 50}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)),
//            new Experiment("12_relu_50_50_l2_decay", new int[]{50, 50}, new TrainParams.Builder()
//                    .batch_size(64)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)
//                    .decay_rate(1 - 1e-2)
//                    .decay_step(1)),
//
//            /* 30-50-50 */
//            new Experiment("13_relu_30_50_30_overfit", new int[]{30, 50, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("14_relu_30_50_30_overfit_normfeat", new int[]{30, 50, 30}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)),
//            new Experiment("15_relu_30_50_30_overfit_normfeat_shflb", new int[]{30, 50, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(100)
//                    .learning_rate(1e-3)),
////            new Experiment("16_relu_30_50_30_l2_long", new int[]{30, 50, 30}, new TrainParams.Builder()
////                    .batch_size(32)
////                    .normalize_features(true)
////                    .shuffle_batches(true)
////                    .epochs_num(100)
////                    .learning_rate(1e-3)
////                    .regularization_coef(1e-4)),
//            new Experiment("17_relu_30_50_30_l2", new int[]{30, 50, 30}, new TrainParams.Builder()
//                    .batch_size(32)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)),
//            new Experiment("18_relu_30_50_30_l2_decay", new int[]{30, 50, 30}, new TrainParams.Builder()
//                    .batch_size(64)
//                    .normalize_features(true)
//                    .shuffle_batches(true)
//                    .epochs_num(80)
//                    .learning_rate(1e-3)
//                    .regularization_coef(1e-4)
//                    .decay_rate(1 - 1e-2)
//                    .decay_step(1))
    };
}
