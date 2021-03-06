package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.utils.Holder;
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
import hr.fer.zemris.utils.logs.*;
import hr.fer.zemris.utils.threading.WorkArbiter;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU6;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;

public class ArchitectureSearchProgram {
    private static ILogger slack = new SlackLogger("lirfu", "slack_webhook.txt");

    public static void main(String[] args) {
        try {
            run();
        } catch (Exception e) {
            slack.e("Exception in architecture search!");
        }
    }

    private static void run() throws Exception {
        Stopwatch stopwatch = new Stopwatch();

        // Parallelzation.
        final WorkArbiter arbiter = new WorkArbiter("W", 1);

        // Create a common instance of train params.
        String experiment_name = "test_previous";
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

        // Popular activations.
        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("relu[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("min[6,relu[x]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("elu[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("selu[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("lrelu[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("threlu[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("swish[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("sigm[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("hsigm[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("tanh[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("htanh[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("rattanh[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("rectanh[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("softmax[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("softplus[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("softsign[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("sin[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("cos[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("exp[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow2[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow3[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("gauss[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("trsin[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("trcos[x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("plu[x]", set))));

        // GP found activations on DPAv4 for byte.
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow2[cos[-[x,4.153574932708071]]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("elu[elu[*[min[1.0,*[min[1.0,hsigm[x]],0.8102254104210314]],sin[x]]]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("pow2[cos[+[1.0,x]]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("*[-[0.0,swish[swish[cos[-[x,0.8949527515835601]]]]],x]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("swish[elu[sin[cos[+[-[x,1.0],pow3[pow3[0.1607274601962161]]]]]]]", set))));

        // GP found activations on DPAv4 for HW.
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("sin[+[min[sin[+[min[x,0.5068783911631836],1.0]],-0.9990098031722499],x]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("*[cos[x],0.9910098031767487]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("relu[*[abs[x],x]]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("*[cos[x],0.9815578450294762]", set))));
//        activations.add(new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("/[-[x,exp[-[x,1.0]]],1.0]", set))));

        // total: 24 functions

        final int skip = 0;
        int i = 0;
        final LinkedList<Pair<Double, String>> top_results = new LinkedList<>();

        for (String[] ds : new String[][]{
                new String[]{
                        "res/AES_Shivam/AES_Shivam_traces_train_byte.csv ; res/AES_Shivam/AES_Shivam_labels_train_byte.csv",
                        "res/AES_Shivam/AES_Shivam_traces_test_byte.csv  ; res/AES_Shivam/AES_Shivam_labels_test_byte.csv"
                },
                new String[]{
                        "res/ASCAD/ASCAD_traces_train_byte.csv ; res/ASCAD/ASCAD_labels_train_byte.csv",
                        "res/ASCAD/ASCAD_traces_test_byte.csv  ; res/ASCAD/ASCAD_labels_test_byte.csv"
                },
                new String[]{
                        "res/Random_delay/Random_delay_traces_train_byte.csv ; res/Random_delay/Random_delay_labels_train_byte.csv",
                        "res/Random_delay/Random_delay_traces_test_byte.csv  ; res/Random_delay/Random_delay_labels_test_byte.csv"
                },
                //                new String[]{
//                        "res/DPAv2/DPAv2_traces_train_byte.csv ; res/DPAv2/DPAv2_labels_train_byte.csv",
//                        "res/DPAv2/DPAv2_traces_test_byte.csv  ; res/DPAv2/DPAv2_labels_test_byte.csv"
//                },
//                new String[]{
//                        "res/DPAv4/DPAv4_traces_train_byte.csv ; res/DPAv4/DPAv4_labels_train_byte.csv",
//                        "res/DPAv4/DPAv4_traces_test_byte.csv  ; res/DPAv4/DPAv4_labels_test_byte.csv"
//                },
        }) {

            slack.i("Experiments for " + ds[0].split("/")[1]);

            for (String architecture : new String[]{
                    /*"fc(100)-fc(100)", "fc(200)-fc(200)", "fc(300)-fc(300)", "fc(400)-fc(400)", "fc(500)-fc(500)",
                    "fc(50)-fc(50)-fc(50)", "fc(100)-fc(100)-fc(100)", "fc(200)-fc(200)-fc(200)",
                    "fc(200)-fc(50)-fc(200)", "fc(200)-fc(100)-fc(200)", "fc(300)-fc(300)-fc(300)",
                    "fc(50)-fc(50)-fc(50)-fc(50)", "fc(100)-fc(100)-fc(100)-fc(100)", "fc(200)-fc(200)-fc(200)-fc(200)"*/
//                    "fc(198)-fc(174)", "fc(313)-fc(18)-fc(141)"
                    "fc(300)-fc(300)", "fc(300)-fc(300)-fc(300)", "fc(300)-fc(300)-fc(300)-fc(300)"
            }) {

                for (IActivation acti : activations) {
                    if (i++ < skip) continue;

                    try {
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
                        new Formatter(sb).format("%-23s  %-15s  %.3f  %.3f  %.3f  %.3f/%.3d  %.3f",
                                architecture,
                                acti.toString().substring(0, Math.min(15, acti.toString().length())),
                                result.accuracy(), result.f1(), result.f1_micro(), result.avg_guess_entropy(), (int) result.max_guess_entropy(), result.top3_accuracy()
                        );
                        String report = sb.toString();

                        // Add, sort and remove worst.
                        top_results.addLast(new Pair<>(result.accuracy() + 10 * result.f1(), report));
                        top_results.sort((a, b) -> (int) (b.getKey() - a.getKey()));
                        if (top_results.size() > 10)
                            top_results.removeLast();

                        slack.i(report);
                    } catch (Exception e) {
                        slack.e("Exception in experiment:\n  -> " + ds[0] + "\n  -> " + architecture + "\n  -> " + acti.toString());
                    }
                }
            }
        }

//        System.out.println("Pending:\n" + arbiter.getStatus());

        arbiter.waitOn(arbiter.getAllFinishedCondition());

        StringBuilder sb = new StringBuilder();
        sb.append("DONE! (").append(Utilities.formatMiliseconds(stopwatch.stop())).append(")");
        sb.append("\nTop " + top_results.size() + " results:");
        sb.append("\nArchitecture             Function         Acc    F1     F1 (micro)  AGE/MGE  Acc_top3");
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
            train_procedure.collectModelActivationsOnTest(model, context, new Triple<>(-7., 7., 101));
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
                    return new Double[]{1e-3, 1e-4, 1e-5};
                }
            },
//            new GridSearch.IModifier<TrainParams>() {
//                @Override
//                public IBuilder<TrainParams> modify(IBuilder<TrainParams> p, Object value) {
//                    return ((TrainParams.Builder) p).seed((Long) value);
//                }
//
//                @Override
//                public Object[] getValues() {
//                    return new Long[]{40L, 41L, 42L, 43L, 44L};
//                }
//            }
    };
}
