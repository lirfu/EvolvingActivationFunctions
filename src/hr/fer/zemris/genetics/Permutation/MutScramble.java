package hr.fer.zemris.genetics.Permutation;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.utils.Util;

import java.util.Random;

public class MutScramble extends Mutation<Permutation> {
    private final float prob;
    private final Random rand;

    public MutScramble(Random random, float probability) {
        rand = random;
        prob = probability;
    }

    @Override
    public void mutate(Permutation genotype) {
        if (!Util.willOccur(rand, prob)) return;

        int i1 = rand.nextInt(genotype.size());
        int i2 = rand.nextInt(genotype.size());

        int min = Math.min(i1, i2);
        int max = Math.max(i1, i2);

        for (int i = min; i <= max; i++) {
            int val = rand.nextInt(max - min + 1) + min; // Random index from bounds (inclusive).

            Integer t = genotype.get(i);
            genotype.set(i, genotype.get(val));
            genotype.set(val, t);
        }
    }
}
