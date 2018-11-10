package hr.fer.zemris.utils;

import hr.fer.zemris.genetics.Genotype;

import java.util.Random;

public class Util {
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

    public static String formatMiliseconds(long miliseconds) {
        if (miliseconds >= 1000L) {
            long seconds = miliseconds / 1000L;
            miliseconds -= seconds * 1000L;

            if (seconds >= 60L) {
                long minutes = seconds / 60L;
                seconds -= minutes * 60L;

                if (minutes >= 60L) {
                    long hours = minutes / 60L;
                    minutes -= hours * 60L;

                    if (hours >= 24L) {
                        long days = hours / 24L;
                        hours -= days * 24L;

                        return days + "d " + hours + "h " + minutes + "m " + seconds + "." + miliseconds + "s";
                    }
                    return hours + "h " + minutes + "m " + seconds + "." + miliseconds + "s";
                }
                return minutes + "m " + seconds + "." + miliseconds + "s";
            }
            return seconds + "." + miliseconds + "s";
        }
        return "0." + (miliseconds / 100) + (miliseconds % 100 / 10) + (miliseconds % 100 % 10) + "s";
    }
}
