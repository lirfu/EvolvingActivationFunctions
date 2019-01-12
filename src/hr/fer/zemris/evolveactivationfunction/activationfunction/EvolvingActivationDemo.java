package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.evolveactivationfunction.*;
import hr.fer.zemris.evolveactivationfunction.nodes.ConstNode;
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
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class EvolvingActivationDemo {
    private static final String DATASET_PATH = "res/noiseless_Karlo/noiseless_all_training_9class.arff";

    public static void main(String[] args) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Random r = new Random(42);
        TreeNodeSet set = new TreeNodeSet(r) {
            @Override
            public TreeNode getNode(String node_name) {
                TreeNode node = super.getNode(node_name);
                if (node == null) {
                    try {
                        Double val = Double.parseDouble(node_name);
                        node = new ConstNode();
                        node.setExtra(val);
                    } catch (NumberFormatException ignore) {
                    }
                }
                return node;
            }
        };
        // Define initializer
        SRGenericInitializer initializer = new SRGenericInitializer(set, 5);
        // Initialize params class for parsing.
        EvolvingActivationParams.initialize(new ISerializable[]{
                new CrxReturnRandom(r), new CrxSRSwapSubtrees(r), new CrxSRSwapConstants(r), new CrxSRMeanConstants(r),
                new CrxSRSwapNodes(r),
                new MutSRInsertTerminal(set, r), new MutSRInsertRoot(set, r), new MutSRReplaceNode(set, r),
                new MutSRSwapOrder(r), new MutSRReplaceSubtree(set, initializer, r), new MutInitialize<>(initializer),
                new MutSRRandomConstantSet(r, 0, 1), new MutSRRandomConstantSetInt(r, 0, 1),
                new MutSRRandomConstantAdd(r, 1), new MutSRRemoveRoot(r), new MutSRRemoveUnary(r)
        });
        // Build or load the params.
        EvolvingActivationParams params;
        Context c;
        if (args.length == 0 || !new File(args[0]).exists()) {
            params = create_params(DATASET_PATH, r, set);
        } else {
            params = StorageManager.loadEvolutionParameters(args[0]);
        }

        // Define node set.
        set.load(TreeNodeSetFactory.build(new Random(params.seed()), params.node_set()));

        // Define the training procedure.
        TrainProcedure train_proc = new TrainProcedure(params);
        c = train_proc.createContext(params.experiment_name());

        // Store params to experiment result folder.
        StorageManager.storeEvolutionParams(params, c);

        /* NEUROEVOLUTION */
        ILogger evo_logger = new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger());
        evo_logger.d("=====> Parameters:\n" + params.serialize());

        SREvaluator evaluator = new SREvaluator(train_proc, params.architecture(), evo_logger, true);

        // Build and run the algorithm.
        Algorithm algo = buildAlgorithm(params, c, train_proc, set, initializer, evaluator, r, evo_logger);
        Genotype[] population = algo.run(new Algorithm.LogParams(false, true));

        /* RESULTS */

        // Retrain best and store results.
        DerivableSymbolicTree best = (DerivableSymbolicTree) algo.getBest();

        // Manual results in case of error.
//        DerivableSymbolicTree best = new DerivableSymbolicTree(SymbolicTree.parse("min[x,sin[gauss[1.0]]]", set));
//        Genotype[] population = {
//                best.setFitness(-0.9312078459207975),
//                new DerivableSymbolicTree(SymbolicTree.parse("-[gauss[max[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[-[gauss[-[sigm[/[tan[tan[/[tan[^3[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]]]]]],x]]],x]],sigm[-[tan[/[tan[^3[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]],x]]]],x]", set)).setFitness(-0.6326136032829283),
//                new DerivableSymbolicTree(SymbolicTree.parse("-[gauss[max[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[-[gauss[-[sigm[/[tan[tan[/[tan[^3[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]]]]]],x]]],x]],sigm[-[gauss[-[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[0.5055265383772962]]]]]]]],x]]]],x]", set)).setFitness(-0.6278754377157357),
//                new DerivableSymbolicTree(SymbolicTree.parse("-[gauss[max[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[-[gauss[-[sigm[/[tan[tan[/[tan[^3[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]]]]]],x]]],x]],sigm[-[gauss[-[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[-4.6417928159410815]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]],x]", set)).setFitness(-0.6111795023258811),
//                new DerivableSymbolicTree(SymbolicTree.parse("-[gauss[max[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[tan[sigm[-[gauss[-[sigm[/[tan[tan[/[tan[^3[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]]]]]],x]]],x]],sigm[-[gauss[-[sigm[/[tan[tan[/[tan[cos[tan[tan[tan[^2[sigm[3.032885906846367]]]]]]],x]]],x]],cos[tan[tan[tan[tan[sigm[3.032885906846367]]]]]]]],x]]]],x]", set)).setFitness(-0.5956458662683258),
//        };

        evo_logger.i("=====> Retraining best: " + best + "  (" + best.getFitness() + ")");
        best.setResult(null);  // Do this for unknown reasons (dl4j serialization error otherwise).

        CommonModel model = evaluator.buildModelFrom(best);

        Pair<ModelReport, INDArray> result = evaluator.evaluateModel(model, StorageManager.createStatsLogger(c));
        train_proc.storeResults(model, c, result);

        List<Genotype> l = Arrays.asList(population);
        l.sort(Comparator.comparing(Genotype::getFitness));

        evo_logger.i("Done!\n");
        evo_logger.i("=====> Final best: \n" + best + "  (" + best.getFitness() + ")");
        evo_logger.i(result.getKey().serialize());

        // Extract tops to display
        int top_num = Math.min(5, l.size());
        evo_logger.i("=====> Top " + top_num + " functions: ");
        DerivableSymbolicTree[] top = new DerivableSymbolicTree[top_num];
        for (int i = 0; i < top_num; i++) {
            top[i] = (DerivableSymbolicTree) l.get(i);
            evo_logger.i("--> f" + (i + 1) + ": " + top[i].serialize() + "  (" + top[i].getFitness() + ")");
        }

        BufferedImage[] imgs = ViewActivationFunction.displayResult(best, top);
        StorageManager.writeImageOfBest(imgs[0], c);
        StorageManager.writeImageOfTop(imgs[1], c);
    }

    private static Algorithm buildAlgorithm(EvolvingActivationParams params, Context c, TrainProcedure proc, TreeNodeSet set, Initializer init, SREvaluator eval, Random r, ILogger log) throws IOException {
        GenerationTabooAlgorithm.Builder b = new GenerationTabooAlgorithm.Builder();
        b.setTabooAttempts(params.taboo_attempts())
                .setTabooSize(params.taboo_size())
                .setElitism(params.isElitism())
                .setMutationProbability(params.mutation_prob())
                .setPopulationSize(params.population_size())

                .setInitializer(init)
                .setEvaluator(eval)
                .setSelector(new RouletteWheelSelector(r))
                .setGenotypeTemplate(new DerivableSymbolicTree(set, null))
                .setStopCondition(params.condition())

                .setLogger(log)
                .setNumberOfWorkers(params.worker_num())
                .setRandom(r);
        for (Crossover crx : params.crossovers())
            b.addCrossover(crx);
        for (Mutation mut : params.mutations())
            b.addMutation(mut);
        return b.build();
    }

    private static EvolvingActivationParams create_params(String dataset, Random r, TreeNodeSet set) {
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

                .addCrossover(new CrxSRSwapSubtrees(r))
                .addCrossover(new CrxSRSwapNodes(r))
                .addCrossover(new CrxSRSwapConstants(r))
                .addCrossover(new CrxSRMeanConstants(r))
                .addCrossover(new CrxReturnRandom(r))

                .addMutation(new MutSRInsertRoot(set, r))
                .addMutation(new MutSRInsertTerminal(set, r))
                .addMutation(new MutSRRandomConstantSet(r, -5, 5))
                .addMutation(new MutSRReplaceNode(set, r))
                .addMutation(new MutSRSwapOrder(r))
                .addMutation(new MutSRRemoveRoot(r))
                .addMutation(new MutSRRemoveUnary(r))

                .addNodeSet(TreeNodeSetFactory.Set.ALL.toString())

                .train_path(dataset)
                .experiment_name("Demo_parameters")
                .architecture(new int[]{30, 30})
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
