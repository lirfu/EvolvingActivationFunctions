package hr.fer.zemris.evolveactivationfunction.programs;

import hr.fer.zemris.evolveactivationfunction.*;
import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.SREvaluator;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.ConstNode;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.experiments.Experiment;
import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.genetics.*;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.genetics.symboregression.SRGenericInitializer;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRMeanConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapNodes;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapSubtrees;
import hr.fer.zemris.genetics.symboregression.mut.*;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Triple;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.*;

public class EvolvingActivationProgram {
    private static final String DATASET_PATH = "<dataset-path>";

    public static void main(String[] args) throws IOException, InterruptedException {
        Random r = new Random(42);

        // Override deserialization of numerical nodes.
        TreeNodeSet set = new TreeNodeSet(r) {
            @Override
            public TreeNode getNode(String node_name) {
                TreeNode node = super.getNode(node_name);
                if (node == null) {
                    try { // Constant node.
                        Double val = Double.parseDouble(node_name);
                        node = new ConstNode();
                        node.setExtra(val);
                    } catch (NumberFormatException ignore) {
                    }
                }
                return node;
            }
        };
        // Define tree initializer.
        SRGenericInitializer tree_init = new SRGenericInitializer(set, 5);
        // Initialize params class for parsing.
        EvolvingActivationParams.initialize(new ISerializable[]{
                new CrxReturnRandom(), new CrxSRSwapSubtrees(), new CrxSRSwapConstants(), new CrxSRMeanConstants(),
                new CrxSRSwapNodes(),
                new MutSRInsertTerminal(set), new MutSRInsertRoot(set), new MutSRReplaceNode(set),
                new MutSRSwapOrder(), new MutSRReplaceSubtree(set, tree_init), new MutInitialize<>(tree_init),
                new MutSRRandomConstantSet(0, 1), new MutSRRandomConstantSetInt(0, 1),
                new MutSRRandomConstantAdd(1), new MutSRRemoveRoot(), new MutSRRemoveUnary()
        });

        // If params not given, create template and exit.
        if (args.length == 0 || !new File(args[0]).exists()) { // Generate the example params.
            EvolvingActivationParams params = create_example_params(DATASET_PATH, r, set);
            Context c = new Context(params.name(), params.experiment_name());
            StorageManager.storeEvolutionParams(params, c);

            System.err.println("Config file not specified!");
            System.err.println("As a result, template config file generated in: "
                    + StorageManager.createExperimentPath(c)
                    + StorageManager.sol_evo_params_name_);
            System.err.println("Before usage, edit the dataset paths!");
            System.err.println("Usage: ./executable <config-file>");
            System.exit(1);
        }
        // Load common params from file.
        EvolvingActivationParams common_params = StorageManager.loadEvolutionParameters(args[0]);

        GridSearch.IModifier<TrainParams>[] mods = common_params.getModifiers();

        if (mods.length == 0) { // Just use parameters.
            run(common_params, set, tree_init, r);
        } else {// Grid search parameters.
            GridSearch<TrainParams> s = new GridSearch<>(common_params.experiment_name());
            Iterable<Experiment<TrainParams>> it = s.buildGridSearchExperiments(
                    new EvolvingActivationParams.Builder().cloneFrom(common_params), mods);
            for (Experiment<TrainParams> experiment : it) {
                ((EvolvingActivationParams) experiment.getParams()).experiment_name(experiment.getName());
                run((EvolvingActivationParams) experiment.getParams(), set, tree_init, r);
            }
        }
    }

    private static void run(EvolvingActivationParams params, TreeNodeSet set, SRGenericInitializer tree_init, Random r) throws IOException, InterruptedException {
        // Define node set.
        set.load(TreeNodeSetFactory.build(new Random(params.seed()), params.node_set()));

        // Define the training procedure.
        ITrainProcedure train_proc = new TrainProcedureDL4J(params);
        Context c = train_proc.createContext(params.experiment_name());

        // Store params to experiment result folder.
        StorageManager.storeEvolutionParams(params, c);

        /* NEUROEVOLUTION */
        ILogger evo_logger = new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger());
        evo_logger.i("===> Timestamp:\n" + new Timestamp(new Date().getTime()));
        evo_logger.d("===> Parameters:\n" + params.serialize());
        evo_logger.i("===> Dataset distributions:\n" + train_proc.describeDatasets());

        SREvaluator evaluator = new SREvaluator(train_proc, params.architecture(), evo_logger, true);

        // Build and run the algorithm.
        Algorithm algo = buildAlgorithm(params, set, tree_init, evaluator, r, evo_logger);

        try {
            algo.run(new Algorithm.LogParams(false, true));
        } catch (NullPointerException e) {
            // Use results from previous iteration (is they exist).
        }

        /* RESULTS */

        // Fetch best result.
        DerivableSymbolicTree best = (DerivableSymbolicTree) algo.getBest();
//        // Manual results in case of error.
//        DerivableSymbolicTree best = new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.);
//        Genotype[] population = {
//                best,
//                new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.),
//                new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.),
//                new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.),
//                new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.),
//                new DerivableSymbolicTree(SymbolicTree.parse("", set)).setFitness(-0.),
//        };

        // Retrain the best model.
        evo_logger.i("===> Retraining best: " + best + "  (" + best.getFitness() + ")");
        best.setResult(null);  // Do this for unknown reasons (dl4j serialization error otherwise).
        Pair<ModelReport, Object> result = evaluator.evaluateModel(best, StorageManager.createStatsLogger(c), best.serialize());
        IModel model = evaluator.buildModelFrom(best);
        train_proc.storeResults(model, c, result);
        evo_logger.i("Done!\n");
        evo_logger.i("===> Final best: \n" + best + "  (" + best.getFitness() + ")");
        evo_logger.i(result.getKey().serialize());

        // Extract tops to display.
        LinkedList<Triple<Long, String, Double>> optima = algo.getResultBundle().getOptimumHistory();
        evo_logger.i("===> Top " + optima.size() + " functions: ");
        DerivableSymbolicTree[] top = new DerivableSymbolicTree[optima.size()];
        for (int i = 0; i < optima.size(); i++) {
            Triple<Long, String, Double> g = optima.get(i);
            top[i] = new DerivableSymbolicTree(SymbolicTree.parse(g.getVal(), set));
            evo_logger.i("--> f" + (i + 1) + ": " + g.getVal() + "  (" + g.getExtra() + ") at iteration " + g.getKey());
        }

        // Create function images.
        BufferedImage[] imgs = ViewActivationFunction.displayResult(best, top);
        StorageManager.writeImageOfBest(imgs[0], c);
        StorageManager.writeImageOfTop(imgs[1], c);
    }

    /**
     * Build an algorithm instance from given objects.
     */
    private static Algorithm buildAlgorithm(EvolvingActivationParams params, TreeNodeSet set, Initializer init,
                                            SREvaluator eval, Random r, ILogger log) throws IOException {
        // Algorithm specific params.
        Algorithm.Builder a;
        switch (params.getAlgorithm()) {
            // TODO GenerationAlgorithm needs some love.
//            case GENERATION:
//                a = new GenerationAlgorithm.Builder();
//                break;
            case GENERATION_TABOO:
                a = new GenerationTabooAlgorithm.Builder();
                ((GenerationTabooAlgorithm.Builder) a)
                        .setTabooAttempts(params.taboo_attempts())
                        .setTabooSize(params.taboo_size())
                        .setElitism(params.isElitism());
                break;
            // TODO EliminationAlgorithm needs some love.
//            case ELIMINATION:
//                a = new EliminationAlgorithm.Builder();
//                break;
            default:
                throw new IllegalArgumentException("Unknown algorithm: " + params.getAlgorithm());
        }
        // Generic algorithm params.
        a.setMutationProbability(params.mutation_prob())
                .setPopulationSize(params.population_size())
                .setTopOptimaNumber(5)

                .setInitializer(init)
                .setEvaluator(eval)
                .setSelector(new RouletteWheelSelector(r))
                .setGenotypeTemplate(new DerivableSymbolicTree(set, null))
                .setStopCondition(params.condition())

                .setLogger(log)
                .setNumberOfWorkers(params.worker_num())
                .setRandom(r);
        for (Crossover crx : params.crossovers())
            a.addCrossover(crx);
        for (Mutation mut : params.mutations())
            a.addMutation(mut);
        return a.build();
    }

    /**
     * Creates an example of parameters.
     */
    private static EvolvingActivationParams create_example_params(String dataset, Random r, TreeNodeSet set) {
        return (EvolvingActivationParams) new EvolvingActivationParams.Builder()
                .elitism(true)
                .mutation_prob(0.3)
                .population_size(20)
                .stop_condition(new StopCondition.Builder()
                        .setMaxIterations(50)
                        .setMinFitness(-1.0)
                        .build())
                .taboo_attempts(3)
                .taboo_size(5)
                .worker_num(4)

                .addCrossover(new CrxSRSwapSubtrees())
                .addCrossover(new CrxSRSwapNodes())
                .addCrossover(new CrxSRSwapConstants())
                .addCrossover(new CrxSRMeanConstants())
                .addCrossover(new CrxReturnRandom())

                .addMutation(new MutSRInsertRoot(set))
                .addMutation(new MutSRInsertTerminal(set))
                .addMutation(new MutSRRandomConstantSet(-5, 5))
                .addMutation(new MutSRReplaceNode(set))
                .addMutation(new MutSRSwapOrder())
                .addMutation(new MutSRRemoveRoot())
                .addMutation(new MutSRRemoveUnary())

                .addNodeSet(TreeNodeSets.ALL.toString())

                .train_path(dataset)
                .test_path(dataset)
                .experiment_name("Demo_parameters")
                .architecture(new NetworkArchitecture("fc(30)-fc(30)"))
                .train_percentage(.8f)
                .seed(42)

                .batch_size(64)
                .normalize_features(true)
                .shuffle_batches(true)
                .epochs_num(10)
                .learning_rate(2e-3)
                .regularization_coef(1e-4)
                .decay_rate(1 - 1e-2)
                .decay_step(1)
                .build();
    }
}
