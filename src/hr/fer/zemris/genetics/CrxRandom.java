package hr.fer.zemris.genetics;

import java.util.Random;

public class CrxRandom<G extends Genotype> extends Crossover<G> {
    private Random r_;

    public CrxRandom(Random random) {
        r_ = random;
    }

    @Override
    public G cross(G parent1, G parent2) {
        return (G) (r_.nextBoolean() ? parent1.copy() : parent2.copy());
    }
}
