package hr.fer.zemris.genetics.Permutation;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.utils.Util;

import java.util.Random;

public class MutSwap extends Mutation<Permutation> {
    private final float prob;
    private final Random rand;

    public MutSwap(Random random, float probability) {
        rand = random;
        prob = probability;
    }

    @Override
    public void mutate(Permutation genotype) {
        if (!Util.willOccur(rand, prob)) return;

        int i1 = rand.nextInt(genotype.size());
        int i2 = rand.nextInt(genotype.size());

        Integer t = genotype.get(i1);
        genotype.set(i1, genotype.get(i2));
        genotype.set(i2, t);
    }
}
