package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.Triple;
import hr.fer.zemris.utils.Utilities;

import java.util.LinkedList;

public class Result {
    private Genotype best;
    private double min;
    private double max;
    private double avg;
    private double relstddev;
    private long iterations;
    private long evaluations;
    private long elapsed_time;
    private LinkedList<Triple<Long, String, Double>> optimumHistory;

    Result(Genotype best, long iterations, long evaluations, double min, double max, double avg, double relstddev, long elapsed_time,
           LinkedList<Triple<Long, String, Double>> optimumHistory) {
        this.best = best;
        this.evaluations = evaluations;
        this.iterations = iterations;
        this.min = min;
        this.max = max;
        this.avg = avg;
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

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }

    public double getAvg() {
        return avg;
    }

    public double getRelStddev() {
        return relstddev;
    }

    public LinkedList<Triple<Long, String, Double>> getOptimumHistory() {
        return optimumHistory;
    }

    @Override
    public String toString() {
        return generateString(best, min, max, avg, relstddev, iterations, evaluations, elapsed_time);
    }

    public static String generateString(Genotype g, double min, double max, double avg, double relstddev, long iterations, long evaluations, long elapsed_time) {
        return g.serialize() +
                "\nFitness: " + g.fitness_ +
                "\nMin: " + min +
                "\nMax: " + max +
                "\nAvg: " + avg +
                "\nRelStdDev: " + relstddev +
                "\nIteration: " + iterations +
                "\nEvaluations: " + evaluations +
                "\nElapsed time: " + Utilities.formatMiliseconds(elapsed_time);
    }
}