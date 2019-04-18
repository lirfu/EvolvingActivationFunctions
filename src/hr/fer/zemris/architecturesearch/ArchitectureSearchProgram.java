package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.nn.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.CosNode;
import hr.fer.zemris.experiments.Experiment;
import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
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
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.LinkedList;
import java.util.Random;

public class ArchitectureSearchProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        String train_ds = "res/noiseless_data/noiseless_all_training_9class.arff";
        String test_ds = "res/noiseless_data/noiseless_all_testing_9class.arff";

//        IActivation common_activation;
//        common_activation = new ActivationReLU();

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
        String experiment_name = "common_functions";
        TrainParams.Builder common_params = new TrainParams.Builder()
                .name(experiment_name)
                .epochs_num(10)
                .batch_size(64)
                .normalize_features(true)
                .shuffle_batches(true)
                .batch_norm(true)
                .decay_rate(0.99)
                .decay_step(1)
                .dropout_keep_prob(1)
                .seed(42);
        String architecture = "fc(30)-fc(30)";

        // Run experiments.
        TreeNodeSet set = TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL);
        LinkedList<IActivation> activations = new LinkedList<>();
        activations.add(new ActivationReLU());
        activations.add(new ActivationReLU6());
        activations.add(new ActivationELU());
        activations.add(new ActivationSELU());
        activations.add(new ActivationLReLU());
        activations.add(new ActivationRReLU());
        activations.add(new ActivationThresholdedReLU());
        activations.add(new ActivationSwish());
        activations.add(new ActivationCube());
        activations.add(new ActivationSigmoid());
        activations.add(new ActivationHardSigmoid());
        activations.add(new ActivationTanH());
        activations.add(new ActivationHardTanH());
        activations.add(new ActivationRationalTanh());
        activations.add(new ActivationRectifiedTanh());
        activations.add(new ActivationSoftmax());
        activations.add(new ActivationSoftPlus());
        activations.add(new ActivationSoftSign());
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("sin[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("cos[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("tan[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("exp[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow2[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow3[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("log[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("gauss[x]", set))));
        for (IActivation acti : activations) {
            // Create grid search experiments.
            Iterable<Experiment<TrainParams>> experiments =
                    new GridSearch<TrainParams>(experiment_name + '_' + architecture + '_' + acti.toString())
                            .buildGridSearchExperiments(common_params, grid_search_modifiers);

            for (Experiment<TrainParams> e : experiments) {
                arbiter.postWork(() -> {
                    try {
                        TrainProcedureDL4J train_procedure = new TrainProcedureDL4J(train_ds, test_ds, new TrainParams.Builder().cloneFrom(e.getParams()));
                        CommonModel model = train_procedure.createModel(new NetworkArchitecture(architecture), new IActivation[]{acti});
                        Context context = train_procedure.createContext(e.getName());

                        ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
                        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);

                        log.d("===> Architecture: " + architecture);
                        log.d("===> Activation function: " + acti.toString());
                        log.d("===> Parameters:");
                        log.d(e.getParams().toString());

                        Stopwatch timer = new Stopwatch();
                        timer.start();
                        train_procedure.train_joined(model, log, stat_storage);
                        Pair<ModelReport, Object> result = train_procedure.test(model);
                        log.d("===> (" + Utilities.formatMiliseconds(timer.stop()) + ") Result:\n" + result.getKey().serialize());
                        train_procedure.storeResults(model, context, result);
//                    train_procedure.displayTrainStats(stat_storage);
                    } catch (IOException | InterruptedException e1) {
                        e1.printStackTrace();
                    }
                });
            }
        }

        System.out.println("Pending:\n" + arbiter.getStatus());

        arbiter.waitOn(arbiter.getFinishedCondition());
    }

    private static final GridSearch.IModifier[] grid_search_modifiers = {
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).regularization_coef((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{/*1e-2, 1e-3,*/ 1e-4};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).learning_rate((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{2e-3 /*1e-2, 5e-3, 1e-3*/};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).epochs_num((Integer) value);
                }

                @Override
                public Object[] getValues() {
                    return new Integer[]{10 /*5, 10, 15*/};
                }
            }
    };
}
