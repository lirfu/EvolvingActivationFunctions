package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.experiments.Experiment;
import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.IBuilder;
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

public class ArchitectureSearchProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        String train_ds = "res/noiseless/5k/9class/noiseless_9class_5k_train.arff";
        String test_ds = "res/noiseless/5k/9class/noiseless_9class_5k_test.arff";

        IActivation common_activation;
        common_activation = new ActivationReLU();

//        // Use custom activation from a tree.
//        DerivableSymbolicTree tree = (DerivableSymbolicTree) new DerivableSymbolicTree.Builder()
//                .setNodeSet(new TreeNodeSet(new Random()))
////                .add(new ReLUNode())
//                .add(new CustomReLUNode())
//                .add(new InputNode())
//                .build();
//        common_activation = new CustomFunction(tree);

        // Paralelization.
        final WorkArbiter arbiter = new WorkArbiter("Experimenter", 3);

        // Create a common instance of train params.
        String experiment_name = "test_gs";
        TrainParams.Builder common_params = new TrainParams.Builder()
                .name(experiment_name)
                .epochs_num(1)
                .learning_rate(0.001)
                .regularization_coef(1e-4);
        String architecture = "fc(30)-fc(30)";

        // Create grid search experiments.
        Iterable<Experiment<TrainParams>> experiments = new GridSearch<TrainParams>(experiment_name)
                .buildGridSearchExperiments(common_params, grid_search_modifiers);

        // Run experiments.
        for (Experiment<TrainParams> e : experiments) {
            arbiter.postWork(() -> {
                try {
                    TrainProcedureDL4J train_procedure = new TrainProcedureDL4J(train_ds, test_ds, new TrainParams.Builder().cloneFrom(e.getParams()));
                    CommonModel model = train_procedure.createModel(new NetworkArchitecture(architecture), new IActivation[]{common_activation});
                    Context context = train_procedure.createContext(e.getName());

                    ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
                    FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
                    Stopwatch timer = new Stopwatch();

                    timer.start();
                    log.d("Training...");
                    train_procedure.train(model, log, stat_storage);
                    log.d("Testing...");
                    Pair<ModelReport, Object> result = train_procedure.test(model);
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

    private static final GridSearch.IModifier[] grid_search_modifiers = {
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).normalize_features((Boolean) value);
                }

                @Override
                public Object[] getValues() {
                    return new Boolean[]{false, true};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).shuffle_batches((Boolean) value);
                }

                @Override
                public Object[] getValues() {
                    return new Boolean[]{false, true};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).batch_norm((Boolean) value);
                }

                @Override
                public Object[] getValues() {
                    return new Boolean[]{false, true};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).regularization_coef((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{0.1, 0.01, 0.001};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).learning_rate((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{0.01, 0.005, 0.001};
                }
            }
    };
}
