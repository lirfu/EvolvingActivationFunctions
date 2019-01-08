package hr.fer.zemris.genetics;

import java.util.ArrayList;
import java.util.Random;

public class Utils {
    /**
     * Returns true if the event will occur given its probability.
     */
    public static boolean willOccur(double probability, Random rand) {
        return rand.nextDouble() < probability;
    }

    /**
     * Returns the highest unit in the list.
     */
    public static <T extends Comparable<T>> T findHighest(T[] list) {
        T best = list[0];
        for (int i = 1; i < list.length; i++)
            if (best.compareTo(list[i]) < 0)
                best = list[i];
        return best;
    }

    /**
     * Returns the lowest unit in the list.
     */
    public static <T extends Comparable<T>> T findLowest(T[] list) {
        T best = list[0];
        for (int i = 1; i < list.length; i++)
            if (best.compareTo(list[i]) > 0)
                best = list[i];
        return best;
    }

    public static double calculateAverage(Genotype[] population) {
        double sum = 0;
        for (Genotype g : population)
            sum += g.getFitness();
        return sum / population.length;
    }

    /**
     * Returns the relative standard deviation of the fitness values in the population.
     */
    public static double calculateRelativeStandardDeviation(Genotype[] population) {
        int size = population.length;
        double mean = 0;
        for (Genotype g : population)
            mean += g.fitness_;
        mean /= size;

        double squares = 0;
        for (Genotype g : population)
            squares += Math.pow(g.fitness_ - mean, 2);

        return Math.sqrt(squares / (size - 1)) / (mean + 1e-32);
    }

    /**
     * Returns a random_ operator. Probability of selecting a particular operator is proportional to its importance value (roulette wheel).
     */
    public static Operator getRandomOperator(ArrayList<? extends Operator> list, Random rand) {
        if (list.size() == 0)
            return null;
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
}
