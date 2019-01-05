package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Utilities;

import java.util.LinkedList;

public class Result {
    private Genotype best;
    private double relstddev;
    private long iterations;
    private long evaluations;
    private long elapsed_time;
    private LinkedList<Pair<Long, Genotype>> optimumHistory;

    Result(Genotype best, long iterations, long evaluations, double relstddev, long elapsed_time, LinkedList<Pair<Long, Genotype>> optimumHistory) {
        this.best = best;
        this.evaluations = evaluations;
        this.iterations = iterations;
        this.relstddev = relstddev;
        this.elapsed_time = elapsed_time;
        this.optimumHistory = optimumHistory;
    }

    public Genotype getBest() {
        return best;
    }

    public long getEvaluations() {
        return evaluations;
    }

    public long getIterations() {
        return iterations;
    }

    public long getElapsedTime() {
        return elapsed_time;
    }

    public double getRelStddev() {
        return relstddev;
    }

    public LinkedList<Pair<Long, Genotype>> getOptimumHistory() {
        return optimumHistory;
    }

    @Override
    public String toString() {
        return generateString(best, relstddev, iterations, evaluations, elapsed_time);
    }

    public static String generateString(Genotype g, double relstddev, long iterations, long evaluations, long elapsed_time) {
        return g.serialize() +
                "\nFitness: " + g.fitness_ +
                "\nRelStdDev: " + relstddev +
                "\nIteration: " + iterations +
                "\nEvaluations: " + evaluations +
                "\nElapsed time: " + Utilities.formatMiliseconds(elapsed_time);
    }
}