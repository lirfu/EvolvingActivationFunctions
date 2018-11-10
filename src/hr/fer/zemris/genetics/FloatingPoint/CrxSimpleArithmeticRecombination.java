package hr.fer.zemris.genetics.FloatingPoint;

import hr.fer.zemris.genetics.Crossover;

import java.util.Random;

public class CrxSimpleArithmeticRecombination extends Crossover<FloatingPoint> {
    private Random random;

    public CrxSimpleArithmeticRecombination(Random rand) {
        this.random = rand;
    }

    @Override
    public FloatingPoint cross(FloatingPoint parent1, FloatingPoint parent2) {
        FloatingPoint child = (FloatingPoint) (random.nextBoolean() ? parent1.copy() : parent2.copy());

        int index;
        if (parent1.size() > 2) index = 1 + random.nextInt(parent1.size() - 2);
        else if(parent1.size() < 2) return child;
        else index = 1;

        for (int i = index; i < child.size(); i++)
            child.set(i, (parent1.get(i) + parent2.get(i)) / 2);

        return child;
    }
}
