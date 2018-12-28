package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.Pair;

import java.util.LinkedList;

public class Result {
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