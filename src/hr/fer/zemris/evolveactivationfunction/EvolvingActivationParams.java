package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.neurology.dl4j.TrainParams;

import java.util.LinkedList;


public class EvolvingActivationParams extends TrainParams {
    private int population_size_;
    private double mutation_prob_;
    private boolean elitism_;
    private int taboo_size_;
    private int taboo_attempts_;
    private LinkedList<Crossover> crossovers_;
    private LinkedList<Mutation> mutations_;

    private int[] architecture_;
    private String activation_;
    private double train_percentage_;
    private String train_path_;
    private String test_path_;

    public EvolvingActivationParams() {
    }

    public EvolvingActivationParams(TrainParams train_params, int population_size, double mutation_prob,
                                    boolean elitism, int taboo_size, int taboo_attempts,
                                    LinkedList<Crossover> crossovers, LinkedList<Mutation> mutations,
                                    int[] architecture, String activation, double train_percentage, String train_path, String test_path) {
        super(train_params);
        population_size_ = population_size;
        mutation_prob_ = mutation_prob;
        elitism_ = elitism;
        taboo_size_ = taboo_size;
        taboo_attempts_ = taboo_attempts;
        crossovers_ = crossovers;
        mutations_ = mutations;
        architecture_ = architecture;
        activation_ = activation;
        train_percentage_ = train_percentage;
        train_path_ = train_path;
        test_path_ = test_path;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder(super.serialize())
                .append("population_size").append('\t').append(population_size_).append('\n')
                .append("mutation_prob").append('\t').append(mutation_prob_).append('\n')
                .append("elitism").append('\t').append(elitism_).append('\n')
                .append("taboo_size").append('\t').append(taboo_size_).append('\n')
                .append("taboo_attempts").append('\t').append(taboo_attempts_).append('\n');
        for (Crossover c : crossovers_) {
            sb.append(c.serialize());
        }
        for (Mutation m : mutations_) {
            sb.append(m.serialize());
        }
        sb.append("architecture");
        for (int i = 0; i < architecture_.length; i++) {
            sb.append(architecture_[i]);
            if (i < architecture_.length - 1)
                sb.append('-');
        }
        sb.append('\n');
        if (activation_ != null)
            sb.append("activation").append('\t').append(activation_).append('\n');
        sb.append("train_percentage").append('\t').append(train_percentage_).append('\n');
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
            case "train_percentage":
                train_percentage_ = Double.parseDouble(parts[1]);
                break;
            case "train_path":
                train_path_ = parts[1];
                break;
            case "test_path":
                test_path_ = parts[1];
                break;
        }
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
        private int[] architecture_;
        private String activation_;
        private double train_percentage_ = 0.8;
        private String train_path_, test_path_;

        public EvolvingActivationParams build() {
            if (architecture_ == null)
                throw new IllegalArgumentException("Network architecture must be defined!");
            if (train_path_ == null)
                throw new IllegalArgumentException("Train dataset path must be specified!");

            return new EvolvingActivationParams(super.build(), population_size_, mutation_prob_,
                    elitism_, taboo_size_, taboo_attempts_, crossovers_, mutations_,
                    architecture_, activation_, train_percentage_, train_path_, test_path_);
        }

        public Builder population_size(int size) {
            population_size_ = size;
            return this;
        }

        public Builder setMutationProb(double probability) {
            mutation_prob_ = probability;
            return this;
        }

        public Builder setElitism(boolean elitism) {
            elitism_ = elitism;
            return this;
        }

        public Builder setTabooSize(int size) {
            taboo_size_ = size;
            return this;
        }

        public Builder setTabooAttempts(int num) {
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

        public Builder setArchitecture(int[] arch) {
            architecture_ = arch;
            return this;
        }

        public Builder setActivation(String activation) {
            activation_ = activation;
            return this;
        }

        public Builder setTrainPercentage(double percentage) {
            train_percentage_ = percentage;
            return this;
        }

        public Builder setTrainDatasetPath(String path) {
            train_path_ = path;
            return this;
        }

        public Builder setTestDatasetPath(String path) {
            test_path_ = path;
            return this;
        }
    }
}
