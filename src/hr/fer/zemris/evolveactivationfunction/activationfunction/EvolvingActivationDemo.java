package hr.fer.zemris.evolveactivationfunction.activationfunction;

import hr.fer.zemris.evolveactivationfunction.*;
import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.CrxReturnRandom;
import hr.fer.zemris.genetics.Mutation;
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
import hr.fer.zemris.utils.logs.DevNullLogger;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Random;

public class EvolvingActivationDemo {
    public static void main(String[] args) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Random r = new Random(42);

        TreeNodeSet set = new TreeNodeSetFactory().build(r,
                TreeNodeSetFactory.Set.ARITHMETICS,
                TreeNodeSetFactory.Set.CONSTANT,
                TreeNodeSetFactory.Set.ACTIVATIONS);

        EvolvingActivationParams params;
        TrainProcedure proc;
        Context c;
        if (args.length == 0) {
            params = (EvolvingActivationParams) new EvolvingActivationParams.Builder()
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
                    .train_path("res/noiseless/5k/9class/noiseless_9class_5k_train.arff")
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
        } else {
            params = StorageManager.loadEvolutionParameters(args[0]);
            System.out.println(params.serialize());
        }

        proc = new TrainProcedure(params);
        c = proc.createContext("test_params");
        StorageManager.storeEvolutionParams(params, c);

        Algorithm algo = buildAlgorithm(params, c, proc, set, r);

        algo.run(new Algorithm.LogParams(false, false));
    }

    private static Algorithm buildAlgorithm(EvolvingActivationParams params, Context c,TrainProcedure proc, TreeNodeSet set, Random r) throws IOException {
        GenerationTabooAlgorithm.Builder b = new GenerationTabooAlgorithm.Builder();
        b.setTabooAttempts(params.taboo_attempts())
                .setTabooSize(params.taboo_size())
                .setElitism(params.isElitism())
                .setMutationProbability(params.mutation_prob())
                .setPopulationSize(params.population_size())

                .setInitializer(new SRGenericInitializer(set, 4))
                .setEvaluator(new SREvaluator(proc, params.architecture(), new DevNullLogger()))
                .setSelector(new RouletteWheelSelector(r))
                .setGenotypeTemplate(new DerivableSymbolicTree(set, null))
                .setStopCondition(params.condition())


                .setLogger(StorageManager.createEvolutionLogger(c))
                .setNumberOfWorkers(params.worker_num())
                .setRandom(r);
        for (Crossover crx : params.crossovers())
            b.addCrossover(crx);
        for (Mutation mut : params.mutations())
            b.addMutation(mut);
        return b.build();
    }
}
