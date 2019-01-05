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


    public EvolvingActivationParams() {
    }

    public EvolvingActivationParams(TrainParams train_params, int population_size, double mutation_prob,
                                    boolean elitism, int taboo_size, int taboo_attempts,
                                    LinkedList<Crossover> crossovers, LinkedList<Mutation> mutations) {
        super(train_params);
        population_size_ = population_size;
        mutation_prob_ = mutation_prob;
        elitism_ = elitism;
        taboo_size_ = taboo_size;
        taboo_attempts_ = taboo_attempts;
        crossovers_ = crossovers;
        mutations_ = mutations;
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
        private double mutation_prob_;
        private boolean elitism_;
        private int taboo_size_;
        private int taboo_attempts_;
        private LinkedList<Crossover> crossovers_ = new LinkedList<>();
        private LinkedList<Mutation> mutations_ = new LinkedList<>();

        public EvolvingActivationParams build() {
            return new EvolvingActivationParams(super.build(), population_size_, mutation_prob_, elitism_, taboo_size_, taboo_attempts_, crossovers_, mutations_);
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
    }
}
