package hr.fer.zemris.genetics;

import hr.fer.zemris.utils.Util;

import java.util.Random;

public class MutOnePoint extends Mutation {
    private Random rand;
    private double probability;

    public MutOnePoint(Random rand, double probability) {
        this.rand = rand;
        this.probability = probability;
    }

    @Override
    public void mutate(Genotype genotype) {
        if (Util.willOccur(rand, probability))
            genotype.set(rand.nextInt(genotype.size()), genotype.generateParameter(rand));
    }
}
