package hr.fer.zemris.evolveheteronet;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.experiments.Experiment;
import hr.fer.zemris.experiments.GridSearch;
import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.vector.VectorGenericInitializer;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.vector.crx.CrxVOnePoint;
import hr.fer.zemris.genetics.vector.crx.CrxVUniform;
import hr.fer.zemris.genetics.vector.intvector.IntVectorGenotype;
import hr.fer.zemris.genetics.vector.mut.MutVGenerateMultiple;
import hr.fer.zemris.genetics.vector.mut.MutVGenerateSingle;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.SlackLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;

import java.io.File;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.Date;
import java.util.Random;

public class EvolveHeteroNetProgram {
    private final static SlackLogger slack = new SlackLogger("Logger", "slack_webhook.txt");


    public static void main(String[] args) throws IOException {
        if (args.length == 0 || !new File(args[0]).exists()) {
            System.err.println("Please provide the evolution params file!");
            System.exit(1);
        }

        // Initialize params class for parsing.
        EvolvingActivationParams.initialize(new ISerializable[]{
                new MutVGenerateSingle(), new MutVGenerateMultiple(5),
                new CrxVOnePoint(), new CrxVUniform()
        });

        // Load common params from file.
        EvolvingActivationParams common_params = StorageManager.loadEvolutionParameters(args[0]);
        // Get modifiers.
        GridSearch.IModifier<TrainParams>[] mods = common_params.getModifiers();

        if (mods.length == 0) { // Just use parameters.
            try {
                run(common_params);
            } catch (Exception e) {
                slack.e("Exception in experiment '" + common_params.name() + "'!");
            }
        } else {// Grid search parameters.
            final int skip = 0;
            int ex_i = 0;

            GridSearch<TrainParams> s = new GridSearch<>(common_params.experiment_name());
            Iterable<Experiment<TrainParams>> it = s.buildGridSearchExperiments(
                    new EvolvingActivationParams.Builder().cloneFrom(common_params), mods);
            for (Experiment<TrainParams> experiment : it) {
                if (ex_i++ < skip) continue;

                ((EvolvingActivationParams) experiment.getParams()).experiment_name(experiment.getName());
                try {
                    run((EvolvingActivationParams) experiment.getParams());
                } catch (Exception e) {
                    slack.e("Exception in experiment '" + experiment.getName() + "'!");
                }
                System.gc();
            }
        }
    }

    private static void run(EvolvingActivationParams params) throws IOException, InterruptedException {
        Random rand = new Random(params.seed());
        TreeNodeSet set = TreeNodeSetFactory.build(rand, params.node_set());

        TrainProcedureDL4J train_proc = new TrainProcedureDL4J(params);
        Context c = train_proc.createContext(params.experiment_name());
        StorageManager.storeEvolutionParams(params, c);

        ILogger log = new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger());
        log.i("===> Timestamp:\n" + new Timestamp(new Date().getTime()));
        log.d("===> Parameters:\n" + params.serialize());
        log.i("===> Dataset distributions:\n" + train_proc.describeDatasets());

        HeteroEvaluator eval = new HeteroEvaluator(train_proc, params.architecture(), log, set);

        Algorithm.Builder a = new GenerationTabooAlgorithm.Builder()
                .setTabooAttempts(params.taboo_attempts())
                .setTabooSize(params.taboo_size())
                .setElitism(params.isElitism())

                .setMutationProbability(params.mutation_prob())
                .setPopulationSize(params.population_size())
                .setTopOptimaNumber(10)

                .setInitializer(new VectorGenericInitializer(rand))
                .setEvaluator(eval)
                .setSelector(new RouletteWheelSelector(rand))
                .setGenotypeTemplate(new IntVectorGenotype(params.architecture().layersNum(), 0, set.getTotalSize() - 1))
                .setStopCondition(params.condition())

                .setLogger(log)
                .setNumberOfWorkers(params.worker_num())
                .setRandom(rand);
        for (Crossover crx : params.crossovers())
            a.addCrossover(crx);
        for (Mutation mut : params.mutations())
            a.addMutation(mut);

        GenerationTabooAlgorithm algo = (GenerationTabooAlgorithm) a.build();
        try {
            algo.run(new Algorithm.LogParams(false, true));
        } catch (Exception e) {
            log.e("GA ended with exception!\n" + e);
//            slack.e("GA ended with exception!");
            throw e;
        }
        System.gc();

        HeteroVectorGenotype best = (HeteroVectorGenotype) algo.getBest();

        log.i("===> Retrain and validate best: " + best + "  (" + best.getFitness() + ")");

        Stopwatch timer = new Stopwatch();
        timer.start();
        IModel m = eval.buildModelFrom(best);
        int best_epoch = train_proc.train_itersearch(m, log, null);
        params.epochs_num(best_epoch);

        m = eval.buildModelFrom(best);
        train_proc.train_joined(m, log, null);
        Pair<ModelReport, Object> result = train_proc.test(m);
        log.i("(" + Utilities.formatMiliseconds(timer.stop()) + ") Done evaluating: " + best.serialize());

        train_proc.storeResults(m, c, result);
        log.i("===> Final best: \n" + best + "  (" + best.getFitness() + ")");
        log.i(result.getKey().toString());
    }
}
