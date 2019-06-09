package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.Holder;
import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.nn.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.experiments.Experiment;
import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.*;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.SlackLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import hr.fer.zemris.utils.threading.WorkArbiter;
import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;

public class ArchitectureSearchProgram {
    private static SlackLogger slack = new SlackLogger("Logger", "slack_webhook.txt");

    public static void main(String[] args) {
        try {
            run();
        } catch (Exception e) {
            slack.e("Exception in architecture search!");
        }
    }

    private static void run() throws Exception {
        Stopwatch stopwatch = new Stopwatch();

        // Paralelization.
        final WorkArbiter arbiter = new WorkArbiter("W", 2);

        // Create a common instance of train params.
        String experiment_name = "common_functions_v2";
        TrainParams.Builder common_params = new TrainParams.Builder()
                .name(experiment_name)
                .epochs_num(40)
                .train_patience(5)
                .convergence_delta(1e-2)
                .batch_size(256)
                .normalize_features(true)
                .shuffle_batches(false)
                .batch_norm(true)
                .decay_rate(0.99)
                .decay_step(1)
                .dropout_keep_prob(1)
                .train_percentage(0.8f)
                .seed(42);

        // Run experiments.
        TreeNodeSet set = TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL);
        LinkedList<IActivation> activations = new LinkedList<>();
//        activations.add(new ActivationReLU());
//        activations.add(new ActivationReLU6());
//        activations.add(new ActivationELU());
//        activations.add(new ActivationSELU());
//        activations.add(new ActivationLReLU());
//        activations.add(new ActivationThresholdedReLU());
//        activations.add(new ActivationSwish());
//        activations.add(new ActivationSigmoid());
//        activations.add(new ActivationHardSigmoid());
//        activations.add(new ActivationTanH());
//        activations.add(new ActivationHardTanH());
//        activations.add(new ActivationRationalTanh());
//        activations.add(new ActivationRectifiedTanh());
//        activations.add(new ActivationSoftmax());
//        activations.add(new ActivationSoftPlus());
//        activations.add(new ActivationSoftSign());
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("sin[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("cos[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("exp[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow2[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow3[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("gauss[x]", set))));
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("trsin[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("trcos[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("plu[x]", set))));


        final LinkedList<Pair<Double, String>> top_results = new LinkedList<>();

        for (String[] ds : new String[][]{
//                new String[]{"res/noiseless_data/noiseless_all_training_9class.arff", "res/noiseless_data/noiseless_all_testing_9class.arff"}
                new String[]{"res/noiseless_data/noiseless_all_training_256class.arff", "res/noiseless_data/noiseless_all_testing_256class.arff"}
        }) {

            for (String architecture : new String[]{
                    "fc(100)-fc(100)", "fc(200)-fc(200)", "fc(300)-fc(300)", "fc(400)-fc(400)", "fc(500)-fc(500)",
                    "fc(50)-fc(50)-fc(50)", "fc(100)-fc(100)-fc(100)", "fc(200)-fc(200)-fc(200)",
                    "fc(200)-fc(50)-fc(200)", "fc(200)-fc(100)-fc(200)",
                    "fc(50)-fc(50)-fc(50)-fc(50)", "fc(100)-fc(100)-fc(100)-fc(100)", "fc(200)-fc(200)-fc(200)-fc(200)"}) {

                for (IActivation acti : activations) {

                    // Create grid search experiments.
                    List<Experiment<TrainParams>> experiments =
                            new GridSearch<TrainParams>(experiment_name + '_' + architecture + '_' + acti.toString())
                                    .buildGridSearchExperiments(common_params, grid_search_modifiers);

                    final Holder<ModelReport> best_result = new Holder<>();
                    final Holder<Experiment<TrainParams>> best_experiment = new Holder<>();
                    final Counter ctr = new Counter(0);

                    for (Experiment<TrainParams> e : experiments) {

                        arbiter.postWork(() -> {
                            try {
                                ModelReport r = run_experiment(ds, architecture, acti, e, true);

                                // Update best result.
                                synchronized (best_result) {
                                    if (!best_result.isDefined()
                                            || best_result.get().f1() < r.f1()
                                            || (best_result.get().f1() == r.f1()
                                            && best_result.get().accuracy() < r.accuracy())) {
                                        best_result.set(r);
                                        best_experiment.set(e);
                                    }
                                }

                                // Update wait condition.
                                ctr.increment();
                            } catch (IOException | InterruptedException exc) {
                                exc.printStackTrace();
                            }
                        });
                    }

                    // Wait for all experiments to finish.
                    arbiter.waitOn(() -> ctr.value() == experiments.size());

                    // Retrain with best hyperparameters.
                    best_experiment.get().setName(best_experiment.get().getName() + "_BEST");
                    ModelReport result = run_experiment(ds, architecture, acti, best_experiment.get(), false);

                    // Update top results.
                    StringBuilder sb = new StringBuilder();
                    new Formatter(sb).format("%-23s  %-15s  %.3f  %.3f  %.3f",
                            architecture,
                            acti.toString().substring(0, Math.min(15, acti.toString().length())),
                            result.accuracy(), result.f1(), result.f1_micro()
                    );

                    // Add, sort and remove worst.
                    top_results.addLast(new Pair<>(result.accuracy() + 10 * result.f1(), sb.toString()));
                    top_results.sort((a, b) -> (int) (b.getKey() - a.getKey()));
                    if (top_results.size() > 10)
                        top_results.removeLast();
                }
            }
        }

//        System.out.println("Pending:\n" + arbiter.getStatus());

        arbiter.waitOn(arbiter.getFinishedCondition());

        StringBuilder sb = new StringBuilder();
        sb.append("DONE! (").append(Utilities.formatMiliseconds(stopwatch.stop())).append(")");
        sb.append("\nTop " + top_results.size() + " results:");
        sb.append("\nArchitecture             Function         Acc    F1     F1 (micro)");
        for (Pair<Double, String> s : top_results)
            sb.append('\n').append(s.getVal());

        System.out.println(sb.toString());
        slack.d(sb.toString());
    }

    private static ModelReport run_experiment(String[] ds, String architecture, IActivation acti, Experiment<TrainParams> e, boolean validating) throws IOException, InterruptedException {
        TrainProcedureDL4J train_procedure = new TrainProcedureDL4J(ds[0], ds[1], new TrainParams.Builder().cloneFrom(e.getParams()));
        CommonModel model = train_procedure.createModel(new NetworkArchitecture(architecture), new IActivation[]{acti});
        Context context = train_procedure.createContext(e.getName());

        ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
//        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
        FileStatsStorage stat_storage = null;
        log.i(train_procedure.describeDatasets());

        log.d("===> Timestamp: " + LocalDateTime.now().toString());
        log.d("===> Architecture: " + architecture);
        log.d("===> Activation function: " + acti.toString());
        log.d("===> Parameters:");
        log.d(e.getParams().toString());

        Stopwatch timer = new Stopwatch();
        timer.start();

        Pair<ModelReport, Object> result;
        if (validating) {
            int opti_epoch = train_procedure.train_itersearch(model, log, stat_storage);
            e.getParams().epochs_num(opti_epoch);  // Set optimal number of epochs.
            result = train_procedure.validate(model);
        } else {
            train_procedure.train_joined(model, log, stat_storage);
            result = train_procedure.test(model);
//            train_procedure.collectModelActivationsOnTest(model, context);  // Very time and memory intensive.
        }

        log.d("===> (" + Utilities.formatMiliseconds(timer.stop()) + ") Result:\n" + result.getKey().toString());
        train_procedure.storeResults(model, context, result);

        return result.getKey();
    }

    private static final GridSearch.IModifier[] grid_search_modifiers = {
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).regularization_coef((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{1e-3, 1e-4, 1e-5};
                }
            },
            new GridSearch.IModifier<TrainParams>() {
                @Override
                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
                    return ((TrainParams.Builder) p).learning_rate((Double) value);
                }

                @Override
                public Object[] getValues() {
                    return new Double[]{1e-3, 5e-4, 1e-4};
                }
            }
    };
}
