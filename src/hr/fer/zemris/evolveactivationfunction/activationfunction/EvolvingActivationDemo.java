package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.evolveactivationfunction.*;
import hr.fer.zemris.evolveactivationfunction.nodes.CustomReLUNode;
import hr.fer.zemris.evolveactivationfunction.nodes.InputNode;
import hr.fer.zemris.genetics.*;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.genetics.symboregression.SRGenericInitializer;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRMeanConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapSubtree;
import hr.fer.zemris.genetics.symboregression.mut.*;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.logging.Logger;

public class EvolvingActivationDemo {
    public static void main(String[] args) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Random r = new Random(42);
        // Build node set.
        TreeNodeSet set = new TreeNodeSetFactory().build(r,
                TreeNodeSetFactory.Set.ARITHMETICS,
                TreeNodeSetFactory.Set.CONSTANT,
                TreeNodeSetFactory.Set.ACTIVATIONS);
        // Define initializer
        Initializer initializer = new SRGenericInitializer(set, 4);
        // Initialize params class for parsing.
        EvolvingActivationParams.initialize(new ISerializable[]{
                new CrxReturnRandom(r), new CrxSRSwapSubtree(r), new CrxSRSwapConstants(r), new CrxSRMeanConstants(r),
                new MutSRInsertTerminal(set, r), new MutSRInsertRoot(set, r), new MutSRReplaceNode(set, r),
                new MutSRSwapOrder(r), new MutSRReplaceSubtree(set, initializer, r), new MutInitialize(initializer),
                new MutSRRandomConstantSet(r, 0, 1), new MutSRRandomConstantSetInt(r, 0, 1),
                new MutSRRandomConstantAdd(r, 1)
        });
        // Build or load the params.
        EvolvingActivationParams params;
        TrainProcedure train_proc;
        Context c;
        if (args.length == 0 || !new File(args[0]).exists()) {
            params = create_params("res/noiseless/5k/9class/noiseless_9class_5k_train.arff", r, set);
        } else {
            params = StorageManager.loadEvolutionParameters(args[0]);
        }
        // Define the training procedure.
        train_proc = new TrainProcedure(params);
        c = train_proc.createContext("test_evo_params");
        // Store if doesn't exist.
        if (args.length == 0 || !new File(args[0]).exists()) StorageManager.storeEvolutionParams(params, c);

        ILogger evo_logger = new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger());
        SREvaluator evaluator = new SREvaluator(train_proc, params.architecture(), evo_logger);

        evo_logger.d("=====> Parameters:\n"+params.serialize());

        // Build and run the algorithm.
        Algorithm algo = buildAlgorithm(params, c, train_proc, set, initializer, evaluator, r, evo_logger);
        Genotype[] population = algo.run(new Algorithm.LogParams(false, false));

        // Retrain best and store results.
        DerivableSymbolicTree best = (DerivableSymbolicTree) algo.getBest();
        evo_logger.i("=====> Retraining best: " + best + "  (" + best.getFitness() + ")");
        best.setResult(null);  // Do this for unknown reasons (dl4j serialization error otherwise).

//        CommonModel model = train_proc.createModel(new int[]{30, 30}, new IActivation[]{new CustomFunction(best.copy())});
//
//        evo_logger.d("Training...");
//        FileStatsStorage stat_storage = StorageManager.createStatsLogger(c);
//        train_proc.train(model, evo_logger, stat_storage);
//        evo_logger.d("Testing...");
//        Pair<ModelReport, INDArray> result = train_proc.test(model);
//        evo_logger.d(result.getKey().toString());
//        train_proc.storeResults(model, c, result);

        CommonModel model = evaluator.buildModelFrom(best);

        Pair<ModelReport, INDArray> result = evaluator.evaluateModel(model, StorageManager.createStatsLogger(c));
        train_proc.storeResults(model, c, result);

        List<Genotype> l = Arrays.asList(population);
        l.sort(Comparator.comparing(Genotype::getFitness));

        evo_logger.i("=====> Final best: \n" + best + "  (" + best.getFitness() + ")");
        // Extract tops to display
        int top_num = Math.min(5, l.size());
        evo_logger.i("=====> Top " + top_num + " functions: ");
        DerivableSymbolicTree[] top = new DerivableSymbolicTree[top_num];
        for (int i = 0; i < top_num; i++) {
            top[i] = (DerivableSymbolicTree) l.get(i);
            evo_logger.i(" - f" + (i + 1) + ": " + top[i].serialize() + "  (" + top[i].getFitness() + ")");
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
                .population_size(10)
                .stop_condition(new StopCondition.Builder().setMaxIterations(50).setMinFitness(-1.0).build())
                .taboo_attempts(3)
                .taboo_size(5)

                .addCrossover(new CrxSRSwapSubtree(r).setImportance(1))
                .addCrossover(new CrxSRSwapConstants(r).setImportance(1))
                .addCrossover(new CrxSRMeanConstants(r).setImportance(1))
                .addCrossover((Crossover) new CrxReturnRandom(r).setImportance(1))

                .addMutation(new MutSRInsertRoot(set, r).setImportance(1))
                .addMutation(new MutSRInsertTerminal(set, r).setImportance(1))
                .addMutation(new MutSRRandomConstantSet(r, -5, 5).setImportance(1))
                .addMutation(new MutSRReplaceNode(set, r).setImportance(1))
                .addMutation(new MutSRSwapOrder(r).setImportance(1))

                .architecture(new int[]{30, 30})
                .train_path(dataset)
                .train_percentage(.8f)

                .batch_size(64)
                .normalize_features(true)
                .shuffle_batches(true)
                .epochs_num(80)

                .learning_rate(1e-3)
                .regularization_coef(1e-4)
                .decay_rate(1 - 1e-2)
                .decay_step(1)
                .build();
    }
}
