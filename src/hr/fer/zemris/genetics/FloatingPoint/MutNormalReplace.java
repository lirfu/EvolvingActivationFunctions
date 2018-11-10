package hr.fer.zemris.genetics.FloatingPoint;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.utils.Util;

import java.util.Random;

public class MutNormalReplace extends Mutation<FloatingPoint> {
    private Random rand;
    private double probability;
    private double stddev;

    public MutNormalReplace(Random random, double probability, double stddev) {
        rand = random;
        this.probability = probability;
        this.stddev = stddev;
    }

    @Override
    public void mutate(FloatingPoint genotype) {
        for (int i = 0; i < genotype.size(); i++)
            if (Util.willOccur(rand, probability))
                genotype.set(i, rand.nextGaussian() * stddev);
    }
}
