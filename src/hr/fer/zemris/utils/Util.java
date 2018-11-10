package hr.fer.zemris.utils;

import com.sun.istack.internal.NotNull;
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

    /**
     * Parses an array of floats from the given string.
     * For supervised data, the label is specified as a single last value.
     * String format e.g.: "v1,v2,v3,v4,l"
     *
     * @param s         String to parse.
     * @param delimiter Regex that splits the string into packets of data.
     */
    public static Pair<float[], Float> parse(@NotNull String s, @NotNull String delimiter) {
        String[] parts = s.split(delimiter);
        int l = parts.length - 1;
        float[] inputs = new float[l];
        float label;

        for (int i = 0; i < l; i++) {
            inputs[i] = Float.parseFloat(parts[i]);
        }
        label = Float.parseFloat(parts[l]);
        return new Pair<>(inputs, label);
    }

    public static String diff(String s1, String s2) {
        StringBuilder sb = new StringBuilder();
        boolean t = false;
        for (int i = 0; i < Math.min(s1.length(), s2.length()); i++) {
            char c1 = s1.charAt(i), c2 = s2.charAt(i);
            if (!t && c1 != c2) {
                sb.append('<');
                t = true;
            } else if (t && c1 == c2) {
                sb.append('>');
                t = false;
            }
            sb.append(c1);
        }
        if (s1.length() != s2.length()) {
            sb.append("xxx...");
        }
        return sb.toString();
    }
}
