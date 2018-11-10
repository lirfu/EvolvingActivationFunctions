package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;

import java.util.Random;

public class RouletteWheelSelector implements Selector {
    @SuppressWarnings("Duplicates")
    @Override
    public Parent[] selectParentsFrom(Random rand, Genotype[] population) {
        int size = population.length;

        double invertedSum = 0;
        for (Genotype g : population)
            invertedSum += 1. / g.getFitness();

        double randomValue1 = rand.nextDouble() * invertedSum;
        Genotype parent1 = null;

        // Select parent 1.
        int index1 = 0;
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += 1 / population[i].getFitness();
            if (sum >= randomValue1) {
                parent1 = population[i];
                index1 = i;
                break;
            }
        }

        if(parent1==null) parent1 = population[size-1];

        double randomValue2 = rand.nextDouble() * (invertedSum - 1 / parent1.getFitness());
        Genotype parent2 = null;

        // Select parent 2.
        int index2 = 0;
        sum = 0;
        for (int i = 0; i < size; i++) {
            if (i == index1) continue;
            sum += 1 / population[i].getFitness();
            if (sum >= randomValue2) {
                parent2 = population[i];
                index2 = i;
                break;
            }
        }

        if (parent2 == null)
            parent2 = population[size - 2];

        return new Parent[]{new Parent(parent1, index1), new Parent(parent2, index2), null};
    }
}
