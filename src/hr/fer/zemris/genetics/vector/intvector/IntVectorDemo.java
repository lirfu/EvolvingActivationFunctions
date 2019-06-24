package hr.fer.zemris.genetics.vector.intvector;

import hr.fer.zemris.genetics.AEvaluator;
import hr.fer.zemris.genetics.Algorithm;
import hr.fer.zemris.genetics.algorithms.GenerationTabooAlgorithm;
import hr.fer.zemris.genetics.selectors.RouletteWheelSelector;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.genetics.vector.VectorGenericInitializer;
import hr.fer.zemris.genetics.vector.crx.CrxVOnePoint;
import hr.fer.zemris.genetics.vector.crx.CrxVUniform;
import hr.fer.zemris.genetics.vector.mut.MutVGenerateSingle;
import hr.fer.zemris.utils.logs.LogLevel;
import hr.fer.zemris.utils.logs.StdoutLogger;

import java.util.Random;

public class IntVectorDemo {
    private final static int vector_size_ = 5;

    private static class Eval extends AEvaluator<IntVectorGenotype> {

        @Override
        public double performEvaluate(IntVectorGenotype g) {
            return Math.abs(g.get(0) - 5)
                    + Math.abs(g.get(1) - 4)
                    + Math.abs(g.get(2) - 3)
                    + Math.abs(g.get(3) - 2)
                    + Math.abs(g.get(4) - 1);
        }
    }

    public static void main(String[] args) {
        Random rand = new Random(42);

        Algorithm a = new GenerationTabooAlgorithm.Builder()
                .setElitism(true)
                .setTabooSize(100)
                .setTabooAttempts(10_000)

                .setPopulationSize(20)
                .setMutationProbability(0.3)

                .setGenotypeTemplate(new IntVectorGenotype(vector_size_, 1, 5))
                .setInitializer(new VectorGenericInitializer(rand))
                .setSelector(new RouletteWheelSelector(rand))
                .setEvaluator(new Eval())
                .setStopCondition(new StopCondition.Builder().setMinFitness(0.).setMaxTimeMs(300_000).build())

                .setTopOptimaNumber(10)
                .setNumberOfWorkers(4)
                .setLogger(new StdoutLogger())
                .setRandom(rand)

                .addCrossover(new CrxVOnePoint())
                .addCrossover(new CrxVUniform())

                .addMutation(new MutVGenerateSingle())

                .build();

        a.run(new Algorithm.LogParams(true, false));

        System.out.println("Best: " + a.getBest().serialize());
    }
}
