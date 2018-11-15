package hr.fer.zemris.genetics;

import java.util.Random;

public class Utils {
    public static boolean willOccur(Random rand, double probability) {
        return rand.nextDouble() < probability;
    }

    public static void sortPopulation(Genotype[] population) {
        for (int i = 0; i < population.length - 1; i++)
            for (int j = i + 1; j < population.length; j++)
                if (population[i].getFitness() > population[j].getFitness()) {
                    Genotype t = population[i];
                    population[i] = population[j];
                    population[j] = t;
                }
    }

    public static Genotype findBest(Genotype[] population) {
        Genotype best = population[0];
        for (int i = 1; i < population.length; i++)
            if (best.getFitness() < population[i].getFitness())
                best = population[i];
        return best;
    }
}
