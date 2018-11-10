package hr.fer.zemris.genetics;

import java.util.Random;

public class CrxUniform extends Crossover {
    private Random random;

    public CrxUniform(Random rand){this.random = rand;}

    @Override
    public Genotype cross(Genotype parent1, Genotype parent2) {
        Genotype child = parent1.copy();

        for (int i = 0; i < child.size(); i++)
            if(random.nextBoolean())
                child.set(i, parent2.get(i));

        return child;
    }
}
