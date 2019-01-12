package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.ISerializable;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

public abstract class Genotype<T> implements Comparable<Genotype<T>>, ISerializable {
    protected double fitness_;

    protected Genotype() {
        fitness_ = Double.MAX_VALUE;
    }

    protected Genotype(Genotype g) {
        fitness_ = g.fitness_;
    }

    public final void evaluate(AEvaluator fitness_func) {
        fitness_ = fitness_func.evaluate(this);
    }

    public final Double getFitness() {
        return fitness_;
    }

    public final Genotype<T> setFitness(double fitness) {
        fitness_ = fitness;
        return this;
    }

    @Override
    public int compareTo(@NotNull Genotype<T> g) {
        return (int) Math.signum(fitness_ - g.fitness_);
    }

    public int compareTo(double fitness) {
        return (int) Math.signum(fitness_ - fitness);
    }

    /* ABSTRACT METHODS */

    public abstract T get(int index);

    public abstract void set(int index, T value);

    public abstract int size();

    public abstract Genotype copy();

    public abstract T generateParameter(Random rand);
}
