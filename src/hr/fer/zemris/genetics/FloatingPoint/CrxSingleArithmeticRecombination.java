package hr.fer.zemris.genetics.FloatingPoint;

import hr.fer.zemris.genetics.Crossover;

import java.util.Random;

public class CrxSingleArithmeticRecombination extends Crossover<FloatingPoint> {
    private Random random;

    public CrxSingleArithmeticRecombination(Random rand) {
        random = rand;
    }

    @Override
    public FloatingPoint cross(FloatingPoint parent1, FloatingPoint parent2) {
        FloatingPoint child = (FloatingPoint) (random.nextBoolean() ? parent1.copy() : parent2.copy());

        int index;
        if (child.size() > 2) index = random.nextInt(child.size());
        else if (child.size() < 2) return child;
        else index = 1;

        child.set(index, (parent1.get(index) + parent2.get(index)) / 2);
        return child;
    }
}
