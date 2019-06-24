package hr.fer.zemris.genetics.vector;

import hr.fer.zemris.genetics.Initializer;

import java.util.Random;

public class VectorGenericInitializer implements Initializer<AVectorGenotype> {
    private Random r_;

    public VectorGenericInitializer(Random rand) {
        r_ = rand;
    }

    @Override
    public void initialize(AVectorGenotype g) {
        for (int i = 0; i < g.size(); i++)
            g.set(i, g.generateParameter(r_));
    }
}
