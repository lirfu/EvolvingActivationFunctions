package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.utils.Utilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Selects parents based on their fitness using the Roulette wheel method.
 * <p><b>IMPORTANT!</b> This method assumes selection probability is proportional to fitness value.</p>
 */
public class RouletteWheelSelector implements Selector {
    private final Random rand_;

    public RouletteWheelSelector(Random random) {
        rand_ = random;
    }

    @Override
    public Parent[] selectParentsFrom(Genotype[] population) {
        // Calculate the total absolute fitness value.
        double accumulator = 0;
        for (Genotype g : population) {
            accumulator += Double.isFinite(g.getFitness()) ? Math.abs(g.getFitness()) : 0;
        }

        if (accumulator == 0) { // If none is found, pick two at random.
            int i1 = rand_.nextInt(population.length), i2 = rand_.nextInt(population.length);
            return new Parent[]{
                    new Parent(population[i1], i1),
                    new Parent(population[i2], i2)
            };
        }

        // Randomly select positions in the sum range.
        double val1 = rand_.nextDouble() * accumulator;
        double val2 = rand_.nextDouble() * accumulator;

        // Search for the positions of selected values and the replacement.
        Parent p1 = null;
        Parent p2 = null;
        accumulator = 0;
        for (int i = 0; i < population.length; i++) {
            accumulator += Double.isFinite(population[i].getFitness()) ? Math.abs(population[i].getFitness()) : 0;
            if (p1 == null && accumulator >= val1) {
                p1 = new Parent(population[i], i);
            }
            if (p2 == null && accumulator >= val2) {
                p2 = new Parent(population[i], i);
            }
        }

        return new Parent[]{p1, p2};
    }

    //    @SuppressWarnings("Duplicates")
//    @Override
//    public Parent[] selectParentsFrom(Genotype[] population) {
//        int size = population.length;
//
//        // FIXME Fitness should rise for better solutions (0 is min). These probabilities do the opposite.
//        double invertedSum = 0;
//        for (Genotype g : population)
//            invertedSum += 1. / g.getFitness();
//
//        double randomValue1 = rand_.nextDouble() * invertedSum;
//        Genotype parent1 = null;
//
//        // Select parent 1.
//        int index1 = 0;
//        double sum = 0;
//        for (int i = 0; i < size; i++) {
//            sum += 1 / population[i].getFitness();
//            if (sum >= randomValue1) {
//                parent1 = population[i];
//                index1 = i;
//                break;
//            }
//        }
//
//        if (parent1 == null) parent1 = population[size - 1];
//
//        double randomValue2 = rand_.nextDouble() * (invertedSum - 1 / parent1.getFitness());
//        Genotype parent2 = null;
//
//        // Select parent 2.
//        int index2 = 0;
//        sum = 0;
//        for (int i = 0; i < size; i++) {
//            if (i == index1) continue;
//            sum += 1 / population[i].getFitness();
//            if (sum >= randomValue2) {
//                parent2 = population[i];
//                index2 = i;
//                break;
//            }
//        }
//
//        if (parent2 == null)
//            parent2 = population[size - 2];
//
//        return new Parent[]{new Parent(parent1, index1), new Parent(parent2, index2), null};
//    }
}
