package hr.fer.zemris.genetics;

import java.util.Random;

public abstract class Genotype<T> implements Comparable<Genotype<T>> {
    protected double fitness_;

    protected Genotype() {
        fitness_ = Double.MAX_VALUE;
    }

    protected Genotype(Genotype g) {
        fitness_ = g.fitness_;
    }

    public final void evaluate(AEvaluator mFitnessFunction) {
        fitness_ = mFitnessFunction.evaluate(this);
    }

    @Override
    public int compareTo(Genotype<T> tGenotype) {
        return (int) Math.signum(fitness_ - tGenotype.fitness_);
    }

    public final Double getFitness() {
        return fitness_;
    }

    public abstract T get(int index);

    public abstract void set(int index, T value);

    public abstract int size();

    public abstract Genotype copy();

    public abstract void initialize(Random rand);

    public abstract String stringify();

    public abstract T generateParameter(Random rand);
}
