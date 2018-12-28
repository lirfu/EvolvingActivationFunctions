package hr.fer.zemris.genetics;

import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import hr.fer.zemris.utils.threading.WorkArbiter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public abstract class Algorithm {
    // Evolutionary algorithm components.
    protected Genotype[] population_;
    protected final Selector selector_;
    protected final AEvaluator evaluator_;
    protected final ArrayList<Crossover> crossover_list_;
    protected final ArrayList<Mutation> mutation_list_;
    protected final double mut_prob_;
    protected final int population_size_;
    private final Initializer initializer_;
    private final StopCondition condition_;
    private final Genotype genotype_template_;
    // Algorithm internal utilities.
    protected final Random random_;
    protected final ILogger log_;
    protected final WorkArbiter work_arbiter_;
    // Algorithm internal variables.
    private long iterations_;
    private long elapsed_time_;
    private final LinkedList<Pair<Long, Genotype>> optimum_history_list_;
    private Genotype best_unit_;
    private long best_iteration_;

    private Algorithm(ArrayList<Crossover> crossovers, ArrayList<Mutation> mutations, Genotype genotype_template,
                      AEvaluator evaluator, Selector selector, Initializer initializer, StopCondition condition, int population_size,
                      double mut_prob, int worker_number, ILogger logger, Random random) {
        selector_ = selector;
        evaluator_ = evaluator;
        initializer_ = initializer;
        crossover_list_ = crossovers;
        mutation_list_ = mutations;
        genotype_template_ = genotype_template;

        condition_ = condition;
        population_size_ = population_size;
        mut_prob_ = mut_prob;

        optimum_history_list_ = new LinkedList<>();
        random_ = random;
        log_ = logger;
        work_arbiter_ = new WorkArbiter("Algorithm", worker_number);
    }

    protected Algorithm(Algorithm a) {
        selector_ = a.selector_;
        evaluator_ = a.evaluator_;
        initializer_ = a.initializer_;
        crossover_list_ = a.crossover_list_;
        mutation_list_ = a.mutation_list_;
        genotype_template_ = a.genotype_template_;

        condition_ = a.condition_;
        population_size_ = a.population_size_;
        mut_prob_ = a.mut_prob_;

        optimum_history_list_ = a.optimum_history_list_;
        random_ = a.random_;
        log_ = a.log_;
        work_arbiter_ = new WorkArbiter(a.work_arbiter_.getName(), a.work_arbiter_.getWorkerNumber());
    }

    /**
     * Runs an iteration of the algorithm on the current population.
     */
    protected abstract void runIteration();

    protected void initializePopulation() {
        population_ = new Genotype[population_size_];
        for (int i = 0; i < population_size_; i++) {
            population_[i] = genotype_template_.copy();
            initializer_.initialize(population_[i]);
            population_[i].evaluate(evaluator_);
        }
    }

    public void run() {
        run(new LogParams());
    }

    public void run(@NotNull LogParams p) {
        // Clear internals.
        iterations_ = 1;
        elapsed_time_ = 0;
        optimum_history_list_.clear();
        best_unit_ = null;
        best_iteration_ = 0;

        initializePopulation();
        if (p.print_population_) {
            log_.d(getPopulationReport());
        }

        // Evaluate population and find the initial best.
        best_unit_ = findBest(population_).copy();
        best_iteration_ = iterations_;

        log_.i("===> Starting algorithm with population of " + population_size_ + " units!\n");
        log_.i(getReport(iterations_, best_unit_));

        long startingTime = System.currentTimeMillis();
        elapsed_time_ = 0;
        // Loop until a condition is satisfied.
        while (!condition_.isSatisfied(getResultBundle())) {
            // Apply operators on this generation.
            runIteration();

            // Check if new best is an improvement.
            Genotype best = findBest(population_);
            boolean improvement = best_unit_.compareTo(best) > 0;
            // Update the global best if needed.
            if (!p.print_improvements_only_ || improvement) {
                best_unit_ = best.copy();
                best_iteration_ = iterations_;
                log_.i(getReport(iterations_, best_unit_));
                if (p.print_population_) {
                    log_.d(getPopulationReport());
                }
            }

            // Update internals;
            optimum_history_list_.add(new Pair<>(iterations_, best.copy()));
            elapsed_time_ = System.currentTimeMillis() - startingTime;
            iterations_++;
        }

        log_.i("\n===> Algorithm ended!");
        log_.i(condition_.report(getResultBundle()));
        log_.i(getReport(--iterations_, best_unit_));
        log_.i("Best iteration: " + best_iteration_);
        if (p.print_population_) {
            log_.d(getPopulationReport());
        }
    }

    private String getReport(long iteration, Genotype best) {
        return "===> Best unit:\n" + best.stringify()
                + "\nFitness: " + best.getFitness()
                + "\nStddev: " + Utils.calculateStandardDeviation(population_)
                + "\nIteration: " + iteration
                + "\nEvaluations: " + evaluator_.getEvaluations()
                + "\nTime: " + Utilities.formatMiliseconds(elapsed_time_);
    }

    private String getPopulationReport() {
        StringBuilder sb = new StringBuilder();
        for (Genotype g : population_) {
            sb
                    .append(g)
                    .append(' ')
                    .append('(')
                    .append(g.fitness_)
                    .append(')')
                    .append('\n');
        }
        return sb.toString();
    }

    /* GETTERS */

    protected Genotype findBest(Genotype[] pop) {
        return Utils.findLowest(pop);
    }

    protected Crossover getRandomCrossover() {
        return (Crossover) Utils.getRandomOperator(crossover_list_, random_);
    }

    protected Mutation getRandomMutation() {
        return (Mutation) Utils.getRandomOperator(mutation_list_, random_);
    }

    public long getIterations() {
        return iterations_;
    }

    public LinkedList<Pair<Long, Genotype>> optimum_history_list() {
        return optimum_history_list_;
    }

    public Genotype getBest() {
        return best_unit_;
    }

    public long getBestIteration() {
        return best_iteration_;
    }

    public Result getResultBundle() {
        return new Result(best_unit_, best_iteration_, evaluator_.getEvaluations(), Utils.calculateStandardDeviation(population_), elapsed_time_, optimum_history_list_);
    }

    /* INTERNAL CLASSES */

    public static class LogParams {
        private boolean print_improvements_only_ = true;
        private boolean print_population_ = false;

        public LogParams() {
        }

        public LogParams(boolean print_improvements_only, boolean print_population) {
            print_improvements_only_ = print_improvements_only;
            print_population_ = print_population;
        }

        public boolean willPrintImprovementsOnly() {
            return print_improvements_only_;
        }

        public boolean willPrintPopulation() {
            return print_population_;
        }
    }

    public static class Builder {
        private ArrayList<Crossover> crossovers_ = new ArrayList<>();
        private ArrayList<Mutation> mutations_ = new ArrayList<>();
        private Genotype genotype_temp_;
        private AEvaluator evaluator_;
        private Integer pop_size_;
        private double mut_prob_ = 0;

        private StopCondition condition_;
        private Selector selector_;
        private Initializer initializer_;

        private ILogger logger_ = new StdoutLogger();
        private int workers_num_ = 1;
        private Random random_ = new Random();

        public Algorithm build() {
            if (genotype_temp_ == null)
                throw new IllegalStateException("Genotype template must be specified!");
            if (condition_ == null)
                throw new IllegalStateException("At least 1 stop condition must be set!");
            if (pop_size_ == null)
                throw new IllegalStateException("Must specify population size!");
            if (selector_ == null)
                throw new IllegalStateException("Must specify a selector!");
            if (initializer_ == null)
                throw new IllegalStateException("Must specify an initializer!");
            if (logger_ == null)
                throw new IllegalStateException("Must specify a logger!");

            return new Algorithm(crossovers_, mutations_, genotype_temp_, evaluator_, selector_, initializer_, condition_, pop_size_, mut_prob_, workers_num_, logger_, random_) {
                @Override
                protected void runIteration() {
                    throw new IllegalStateException("Method runIteration() not implemented!");
                }
            };
        }

        public Builder setGenotypeTemplate(@NotNull Genotype template) {
            genotype_temp_ = template;
            return this;
        }

        public Builder setPopulationSize(int size) {
            pop_size_ = size;
            return this;
        }

        public Builder setMutationProbability(double probability) {
            mut_prob_ = probability;
            return this;
        }

        public Builder setStopCondition(StopCondition condition) {
            condition_ = condition;
            return this;
        }

        public Builder setEvaluator(@NotNull AEvaluator evaluator) {
            evaluator_ = evaluator;
            return this;
        }

        public Builder setSelector(@NotNull Selector selector) {
            selector_ = selector;
            return this;
        }

        public Builder setInitializer(@NotNull Initializer initializer) {
            initializer_ = initializer;
            return this;
        }

        public Builder addCrossover(@NotNull Crossover crx) {
            crossovers_.add(crx);
            return this;
        }

        public Builder addMutation(@NotNull Mutation mut) {
            mutations_.add(mut);
            return this;
        }

        public Builder setLogger(@NotNull ILogger logger) {
            logger_ = logger;
            return this;
        }

        public Builder setNumberOfWorkers(int number) {
            workers_num_ = number;
            return this;
        }

        public Builder setRandom(Random random) {
            random_ = random;
            return this;
        }
    }
}
