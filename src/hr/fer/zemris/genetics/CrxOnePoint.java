package hr.fer.zemris.genetics;

import java.util.Random;

public class CrxOnePoint extends Crossover {
    private Random random;

    public CrxOnePoint(Random rand) {
        this.random = rand;
    }

    @Override
    public Genotype cross(Genotype parent1, Genotype parent2) {
        Genotype child = parent1.copy();

        int index;
        if (child.size() > 2) index = random.nextInt(child.size());
        else if (child.size() < 2) return child;
        else index = 1;

        for (int i = index; i < child.size(); i++)
            child.set(i, parent2.get(i));

        return child;
    }
}
