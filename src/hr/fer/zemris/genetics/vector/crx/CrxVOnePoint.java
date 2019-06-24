package hr.fer.zemris.genetics.vector.crx;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.vector.AVectorGenotype;

public class CrxVOnePoint extends Crossover<AVectorGenotype> {
    @Override
    public AVectorGenotype cross(AVectorGenotype parent1, AVectorGenotype parent2) {
        AVectorGenotype c1 = (AVectorGenotype) parent1.copy();
        AVectorGenotype c2 = (AVectorGenotype) parent2.copy();

        int index = r_.nextInt(c1.size() - 2) + 1;
        for (int i = index; i < c1.size(); i++) {
            Object t = c1.get(i);
            c1.set(i, c2.get(i));
            c2.set(i, t);
        }

        return r_.nextBoolean() ? c1 : c2;
    }

    @Override
    public String getName() {
        return "crx.vec.onepoint";
    }
}
