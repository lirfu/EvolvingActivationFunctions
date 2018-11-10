package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Util;
import hr.fer.zemris.utils.threading.WorkArbiter;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class Algorithm {
    protected final Random random = new Random();
    protected final AFitnessFunction mFitnessFunction;
    protected Selector mSelector;

    private final ArrayList<Crossover> mCrossovers;
    private final ArrayList<Mutation> mMutations;
    private final Genotype mGenotypeTemplate;
    private final int mPopulationSize;

    private final Long mMaxIterations;
    private Long mMaxEvalsCondition;
    private Long mMaxTimeCondition;
    private final Double mMinFitness;

    private long iterations;
    private long mElapsedTime;
    private LinkedList<Pair<Long, Genotype>> optimumHistory;
    private Genotype mBestUnit;
    private long mBestIteration;
    private Genotype[] mPopulation;

    protected WorkArbiter mWorkArbiter;

    private Algorithm(ArrayList<Crossover> crossovers, ArrayList<Mutation> mutations, Genotype genotypeTemplate, AFitnessFunction function, Selector selector, int popSize, Long maxIterCondition, Long maxEvalCondition, Long maxTimeCondition, Double minimumFitnessCondition, Integer workerNumber) {
        mCrossovers = crossovers;
        mMutations = mutations;
        mFitnessFunction = function;
        mGenotypeTemplate = genotypeTemplate;
        mSelector = selector;
        mPopulationSize = popSize;

        mMaxIterations = maxIterCondition;
        mMaxEvalsCondition = maxEvalCondition;
        mMaxTimeCondition = maxTimeCondition;
        mMinFitness = minimumFitnessCondition;

        mWorkArbiter = new WorkArbiter(workerNumber);
    }

    protected Algorithm(Algorithm algorithm) {
        mCrossovers = algorithm.mCrossovers;
        mMutations = algorithm.mMutations;
        mFitnessFunction = algorithm.mFitnessFunction;
        mSelector = algorithm.mSelector;
        mGenotypeTemplate = algorithm.mGenotypeTemplate;
        mPopulationSize = algorithm.mPopulationSize;

        mMaxIterations = algorithm.mMaxIterations;
        mMaxEvalsCondition = algorithm.mMaxEvalsCondition;
        mMaxTimeCondition = algorithm.mMaxTimeCondition;
        mMinFitness = algorithm.mMinFitness;

        mWorkArbiter = new WorkArbiter(algorithm.mWorkArbiter.getWorkerNumber());
    }

    public void run(boolean showTrace) {
        iterations = 0;
        optimumHistory = new LinkedList<>();

        // Initialize population.
        mPopulation = new Genotype[mPopulationSize];
        final Random rand = new Random();
        for (int i = 0; i < mPopulationSize; i++) {
            mPopulation[i] = mGenotypeTemplate.copy();
            mPopulation[i].randomize(rand);
            mPopulation[i].evaluate(mFitnessFunction);
        }

        // Evaluate population and find the initial best.
        mBestUnit = mPopulation[0];
        for (Genotype g : mPopulation)
            if (mBestUnit.getFitness() > g.getFitness())
                mBestUnit = g;
        mBestUnit = mBestUnit.copy();
        mBestIteration = iterations;

        if (showTrace) {
            System.out.println("===> Starting algorithm with population of " + mPopulationSize + " units!\n");
            print(iterations, mBestUnit);
        }

        long startingTime = System.currentTimeMillis();
        mElapsedTime = 0;
        while ((mMaxIterations == null || mMaxIterations > iterations) &&
                (mMinFitness == null || mMinFitness < mBestUnit.getFitness()) &&
                (mMaxEvalsCondition == null || mMaxEvalsCondition > mFitnessFunction.getEvaluations()) &&
                (mMaxTimeCondition == null || mMaxTimeCondition > mElapsedTime)) {
            iterations++;

            // Apply operators on this generation.
            runIteration(mPopulation);

            // Find the best in the population.
            Genotype best = mPopulation[0];
            for (int i = 1; i < mPopulationSize; i++)
                if (best.getFitness() > mPopulation[i].getFitness())
                    best = mPopulation[i];

            mElapsedTime = System.currentTimeMillis() - startingTime;

            // Update the global best if needed.
            if (mBestUnit.getFitness() > best.getFitness()) {
                mBestUnit = best.copy();
                mBestIteration = iterations;
                if (showTrace) {
                    print(iterations, mBestUnit);
                }
            }

            optimumHistory.add(new Pair<>(iterations, best.copy()));
        }

        if (showTrace) {
            System.out.println("\n===> Algorithm ended!");
            if (mMaxIterations != null && mMaxIterations <= iterations) {
                System.out.println("Max iterations achieved!");
            }
            if (mMinFitness != null && mMinFitness >= mBestUnit.getFitness()) {
                System.out.println("Min fitness achieved!");
            }
            if (mMaxEvalsCondition != null && mMaxEvalsCondition <= mFitnessFunction.getEvaluations()) {
                System.out.println("Max evaluations achieved!");
            }
            if (mMaxTimeCondition != null && mMaxTimeCondition <= mElapsedTime) {
                System.out.println("Max time achieved!");
            }
        }

        // Tell listeners that algorithm ended.
        notifyAll();
    }

    public static double standardDeviation(Genotype[] population) {
        int size = population.length;
        double mean = 0;
        for (Genotype g : population)
            mean += g.fitness_;
        mean /= size;

        double squares = 0;
        for (Genotype g : population)
            squares += Math.pow(g.fitness_ - mean, 2);

        return Math.sqrt(squares / (size - 1));
    }

    public Result getResultBundle() {
        return new Result(mBestUnit, mBestIteration, mFitnessFunction.getEvaluations(), standardDeviation(mPopulation), mElapsedTime, optimumHistory);
    }

    private void print(long iterations, Genotype best) {
        System.out.println("\n===> Best unit: " + best.stringify());
        System.out.println("Fitness: " + best.getFitness());
        System.out.println("Stddev: " + standardDeviation(mPopulation));
        System.out.println("Iteration: " + iterations);
        System.out.println("Evaluations: " + mFitnessFunction.getEvaluations());
        System.out.println("Time: " + Util.formatMiliseconds(mElapsedTime));
    }

    protected void runIteration(Genotype[] population) {
        // must be overwritten
    }

    protected Crossover getCrossover(Random rand) {
        return (Crossover) getRandomFrom(mCrossovers, rand);
    }

    protected Mutation getMutation(Random rand) {
        return (Mutation) getRandomFrom(mMutations, rand);
    }

    private Operator getRandomFrom(ArrayList<? extends Operator> list, Random rand) {
        if (list.size() == 1)
            return list.get(0);

        int importanceSum = 0;
        for (Operator o : list)
            importanceSum += o.getImportance();

        int randomSum = rand.nextInt(importanceSum);
        int sum = 0;
        int i;
        for (i = 0; i < list.size() - 1 && sum < randomSum; i++)
            sum += list.get(i).getImportance();
        return list.get(i);
    }

    public long getIterations() {
        return iterations;
    }

    public LinkedList<Pair<Long, Genotype>> getOptimumHistory() {
        return optimumHistory;
    }

    public Genotype getBest() {
        return mBestUnit;
    }

    public long getBestIteration() {
        return mBestIteration;
    }

    public static class Result {
        private long iteration;
        private double stddev;
        private long elapsedTime;
        private LinkedList<Pair<Long, Genotype>> optimumHistory;
        private Genotype best;
        private long evaluations;

        Result(Genotype best, long iteration, long evaluations, double stddev, long elapsedTime, LinkedList<Pair<Long, Genotype>> optimumHistory) {
            this.best = best;
            this.evaluations = evaluations;
            this.iteration = iteration;
            this.stddev = stddev;
            this.elapsedTime = elapsedTime;
            this.optimumHistory = optimumHistory;
        }

        public Genotype getBest() {
            return best;
        }

        public long getEvaluations() {
            return evaluations;
        }

        public long getIteration() {
            return iteration;
        }

        public long getElapsedTime() {
            return elapsedTime;
        }

        public double getStddev() {
            return stddev;
        }

        public LinkedList<Pair<Long, Genotype>> getOptimumHistory() {
            return optimumHistory;
        }

        @Override
        public String toString() {
            return best.stringify() +
                    "\nFitness: " + best.fitness_ +
                    "\nEvaluations: " + evaluations +
                    "\nIteration: " + iteration +
                    "\nElapsed time: " + (elapsedTime / 1000.) + "s";
        }
    }

    public static class Builder {
        private ArrayList<Crossover> mCrossovers = new ArrayList<>();
        private ArrayList<Mutation> mMutations = new ArrayList<>();
        private Genotype mGenotypeTemplate;
        private Integer mPopulationSize;

        private Long mMaxIterCondition;
        private Double mMinimumFitnessCondition;
        private AFitnessFunction mFitnessFunction;
        private Long mMaxEvalsCondition;
        private Long mMaxTimeCondition;
        private Selector mSelector;

        private Integer mThreadNumber = 1;

        public Algorithm create() {
            if (mGenotypeTemplate == null)
                throw new IllegalStateException("Genotype template must be specified!");
            if (mCrossovers.isEmpty())
                throw new IllegalStateException("At least 1 crossover must be set!");
            if (mMaxIterCondition == null && mMinimumFitnessCondition == null && mMaxEvalsCondition == null && mMaxTimeCondition == null)
                throw new IllegalStateException("At least 1 stop condition must be set!");
            if (mPopulationSize == null)
                throw new IllegalStateException("Must specify population size!");
            if (mSelector == null)
                throw new IllegalStateException("Must specify a selector!");
            return new Algorithm(mCrossovers, mMutations, mGenotypeTemplate, mFitnessFunction, mSelector, mPopulationSize, mMaxIterCondition, mMaxEvalsCondition, mMaxTimeCondition, mMinimumFitnessCondition, mThreadNumber);
        }

        public Builder setGenotypeTemplate(Genotype template) {
            mGenotypeTemplate = template;
            return this;
        }

        public Builder setPopulationSize(Integer size) {
            mPopulationSize = size;
            return this;
        }

        public Builder setMaxIterationsCondition(Long maxIterations) {
            mMaxIterCondition = maxIterations;
            return this;
        }

        public Builder setMinFitnessCondition(Double fitness) {
            this.mMinimumFitnessCondition = fitness;
            return this;
        }

        public Builder setMaxEvalsCondition(Long evaluations) {
            this.mMaxEvalsCondition = evaluations;
            return this;
        }

        public Builder setMaxTimeCondition(Long miliseconds) {
            this.mMaxTimeCondition = miliseconds;
            return this;
        }

        public Builder setFitnessFunction(AFitnessFunction function) {
            mFitnessFunction = function;
            return this;
        }

        public Builder setSelector(Selector selector) {
            mSelector = selector;
            return this;
        }

        public Builder addCrossover(Crossover crx) {
            mCrossovers.add(crx);
            return this;
        }

        public Builder addMutation(Mutation mut) {
            mMutations.add(mut);
            return this;
        }

        public Builder setThreadsNumber(Integer number) {
            mThreadNumber = number;
            return this;
        }
    }
}
