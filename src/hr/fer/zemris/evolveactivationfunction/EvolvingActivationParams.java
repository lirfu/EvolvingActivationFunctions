package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.stopconditions.StopCondition;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.neurology.dl4j.TrainParams;

import java.util.LinkedList;


public class EvolvingActivationParams extends TrainParams {
    private int population_size_;
    private double mutation_prob_;
    private boolean elitism_;
    private int taboo_size_;
    private int taboo_attempts_;
    private LinkedList<Crossover> crossovers_ = new LinkedList<>();
    private LinkedList<Mutation> mutations_ = new LinkedList<>();
    private StopCondition condition_ = new StopCondition();
    private int worker_num_;

    private int[] architecture_;
    private String activation_;
    private String train_path_;
    private String test_path_;

    public EvolvingActivationParams() {
    }

    public EvolvingActivationParams(TrainParams train_params, int population_size, double mutation_prob,
                                    boolean elitism, int taboo_size, int taboo_attempts,
                                    LinkedList<Crossover> crossovers, LinkedList<Mutation> mutations,
                                    StopCondition condition, int worker_num,
                                    int[] architecture, String activation, String train_path, String test_path) {
        super(train_params);
        population_size_ = population_size;
        mutation_prob_ = mutation_prob;
        elitism_ = elitism;
        taboo_size_ = taboo_size;
        taboo_attempts_ = taboo_attempts;
        crossovers_ = crossovers;
        mutations_ = mutations;
        condition_ = condition;
        worker_num_ = worker_num;
        architecture_ = architecture;
        activation_ = activation;
        train_path_ = train_path;
        test_path_ = test_path;
    }

    public int population_size() {
        return population_size_;
    }

    public double mutation_prob() {
        return mutation_prob_;
    }

    public boolean isElitism() {
        return elitism_;
    }

    public int taboo_size() {
        return taboo_size_;
    }

    public int taboo_attempts() {
        return taboo_attempts_;
    }

    public LinkedList<Crossover> crossovers() {
        return crossovers_;
    }

    public LinkedList<Mutation> mutations() {
        return mutations_;
    }

    public StopCondition condition() {
        return condition_;
    }

    public int worker_num() {
        return worker_num_;
    }

    public int[] architecture() {
        return architecture_;
    }

    public String activation() {
        return activation_;
    }

    public String train_path() {
        return train_path_;
    }

    public String test_path() {
        return test_path_;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder(super.serialize())
                .append("population_size").append('\t').append(population_size_).append('\n')
                .append("mutation_prob").append('\t').append(mutation_prob_).append('\n')
                .append("elitism").append('\t').append(elitism_).append('\n')
                .append("taboo_size").append('\t').append(taboo_size_).append('\n')
                .append("taboo_attempts").append('\t').append(taboo_attempts_).append('\n')
                .append("worker_num").append('\t').append(worker_num_).append('\n');
        if (crossovers_ != null)
            for (Crossover c : crossovers_)
                sb.append(c.serialize());
        if (mutations_ != null)
            for (Mutation m : mutations_)
                sb.append(m.serialize());
        if (condition_ != null)
            sb.append(condition_.serialize());
        if (architecture_ != null) {
            sb.append("architecture").append('\t');
            for (int i = 0; i < architecture_.length; i++) {
                sb.append(architecture_[i]);
                if (i < architecture_.length - 1)
                    sb.append('-');
            }
            sb.append('\n');
        }
        if (activation_ != null)
            sb.append("activation").append('\t').append(activation_).append('\n');
        sb.append("train_path").append('\t').append(train_path_).append('\n');
        if (test_path_ != null)
            sb.append("test_path").append('\t').append(test_path_).append('\n');
        return sb.toString();
    }

    @Override
    public void parse(String line) {
        super.parse(line);
        String[] parts = line.split(SPLIT_REGEX);
        switch (parts[0]) {
            case "population_size":
                population_size_ = Integer.parseInt(parts[1]);
                break;
            case "mutation_prob":
                mutation_prob_ = Double.parseDouble(parts[1]);
                break;
            case "elitism":
                elitism_ = Boolean.parseBoolean(parts[1]);
                break;
            case "taboo_size":
                taboo_size_ = Integer.parseInt(parts[1]);
                break;
            case "taboo_attempts":
                taboo_attempts_ = Integer.parseInt(parts[1]);
                break;
            case "worker_num":
                worker_num_ = Integer.parseInt(parts[1]);
                break;
            case "architecture":
                String[] p = parts[1].split("-");
                architecture_ = new int[p.length];
                for (int i = 0; i < p.length; i++) {
                    architecture_[i] = Integer.parseInt(p[i]);
                }
                break;
            case "activation":
                activation_ = parts[1];
                break;
            case "train_path":
                train_path_ = parts[1];
                break;
            case "test_path":
                test_path_ = parts[1];
                break;
        }
        condition_.parse(line);
        for (Crossover c : crossovers_) {
            c.parse(line);
        }
        for (Mutation m : mutations_) {
            m.parse(line);
        }
    }

    public static class Builder extends TrainParams.Builder {
        private int population_size_;
        private double mutation_prob_ = 0.1;
        private boolean elitism_ = true;
        private int taboo_size_;
        private int taboo_attempts_;
        private LinkedList<Crossover> crossovers_ = new LinkedList<>();
        private LinkedList<Mutation> mutations_ = new LinkedList<>();
        private StopCondition condition_;
        private int[] architecture_;
        private String activation_;
        private String train_path_, test_path_;
        private int worker_num_ = 1;

        public EvolvingActivationParams build() {
            if (architecture_ == null)
                throw new IllegalStateException("Network architecture must be defined!");
            if (train_path_ == null)
                throw new IllegalStateException("Train dataset path must be specified!");
            if (condition_ == null)
                throw new IllegalStateException("Stop condition must be specified!");

            return new EvolvingActivationParams(super.build(), population_size_, mutation_prob_,
                    elitism_, taboo_size_, taboo_attempts_, crossovers_, mutations_, condition_, worker_num_,
                    architecture_, activation_, train_path_, test_path_);
        }

        public Builder population_size(int size) {
            population_size_ = size;
            return this;
        }

        public Builder mutation_prob(double probability) {
            mutation_prob_ = probability;
            return this;
        }

        public Builder stop_condition(StopCondition condtion) {
            condition_ = condtion;
            return this;
        }

        public Builder elitism(boolean elitism) {
            elitism_ = elitism;
            return this;
        }

        public Builder taboo_size(int size) {
            taboo_size_ = size;
            return this;
        }

        public Builder taboo_attempts(int num) {
            taboo_attempts_ = num;
            return this;
        }

        public Builder addCrossover(Crossover c) {
            crossovers_.add(c);
            return this;
        }

        public Builder addMutation(Mutation m) {
            mutations_.add(m);
            return this;
        }

        public Builder architecture(int[] arch) {
            architecture_ = arch;
            return this;
        }

        public Builder activation(String activation) {
            activation_ = activation;
            return this;
        }

        public Builder train_path(String path) {
            train_path_ = path;
            return this;
        }

        public Builder test_path(String path) {
            test_path_ = path;
            return this;
        }

        public Builder worker_num(int num) {
            worker_num_ = num;
            return this;
        }
    }
}
