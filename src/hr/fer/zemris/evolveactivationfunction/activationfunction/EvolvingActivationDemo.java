package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.evolveactivationfunction.*;
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
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.Random;

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
        Initializer init = new SRGenericInitializer(set, 4);
        // Initialize params class for parsing.
        EvolvingActivationParams.initialize(new ISerializable[]{
                new CrxReturnRandom(r), new CrxSRSwapSubtree(r), new CrxSRSwapConstants(r), new CrxSRMeanConstants(r),
                new MutSRInsertTerminal(set, r), new MutSRInsertRoot(set, r), new MutSRReplaceNode(set, r),
                new MutSRSwapOrder(r), new MutSRReplaceSubtree(set, init, r), new MutInitialize(init),
                new MutSRRandomConstantSet(r, 0, 1), new MutSRRandomConstantSetInt(r, 0, 1),
                new MutSRRandomConstantAdd(r, 1)
        });
        // Build or load the params.
        EvolvingActivationParams params;
        TrainProcedure proc;
        Context c;
        if (args.length == 0 || !new File(args[0]).exists()) {
            params = create_params("res/noiseless/5k/9class/noiseless_9class_5k_train.arff", r, set);
        } else {
            params = StorageManager.loadEvolutionParameters(args[0]);
            System.out.println(params.serialize());
        }
        // Define the training procedure.
        proc = new TrainProcedure(params);
        c = proc.createContext("test_params");
        // Store if doesn't exist.
        if (args.length == 0 || !new File(args[0]).exists()) StorageManager.storeEvolutionParams(params, c);

        // Build and run the algorithm.
        Algorithm algo = buildAlgorithm(params, c, proc, set, init, r);
        algo.run(new Algorithm.LogParams(false, false));

        DerivableSymbolicTree best = (DerivableSymbolicTree) algo.getBest();
        System.out.println("=====> Final best: " + best);

        // Get history of optima.
        LinkedList<Pair<Long, Genotype>> l = algo.getResultBundle().getOptimumHistory();
        l.sort(Comparator.comparing(p -> p.getVal().getFitness()));

        // Extract top to display
        int top_num = 5;
        System.out.println("=====> Top " + top_num + "functions: ");
        SymbolicTree[] top = new SymbolicTree[top_num];
        for (int i = 1; i <= top_num; i++) {
            top[i] = (SymbolicTree) l.get(l.size() - i).getVal();
            System.out.println(" - f" + i + ": " + top[i].serialize());
        }

        ViewActivationFunction.displayResult(best, top);
    }

    private static Algorithm buildAlgorithm(EvolvingActivationParams params, Context c, TrainProcedure proc, TreeNodeSet set, Initializer init, Random r) throws IOException {
        GenerationTabooAlgorithm.Builder b = new GenerationTabooAlgorithm.Builder();
        b.setTabooAttempts(params.taboo_attempts())
                .setTabooSize(params.taboo_size())
                .setElitism(params.isElitism())
                .setMutationProbability(params.mutation_prob())
                .setPopulationSize(params.population_size())

                .setInitializer(init)
                .setEvaluator(new SREvaluator(proc, params.architecture(), new StdoutLogger()))
                .setSelector(new RouletteWheelSelector(r))
                .setGenotypeTemplate(new DerivableSymbolicTree(set, null))
                .setStopCondition(params.condition())

                .setLogger(new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger()))
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
                .stop_condition(new StopCondition.Builder().setMaxIterations(50).setMinFitness(0).build())
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
