package hr.fer.zemris.genetics.vector.crx;

import hr.fer.zemris.genetics.Crossover;
import hr.fer.zemris.genetics.vector.AVectorGenotype;

public class CrxVUniform extends Crossover<AVectorGenotype> {
    @Override
    public AVectorGenotype cross(AVectorGenotype parent1, AVectorGenotype parent2) {
        AVectorGenotype c1 = (AVectorGenotype) parent1.copy();
        AVectorGenotype c2 = (AVectorGenotype) parent2.copy();

        for (int i = 0; i < c1.size(); i++) {
            if (r_.nextBoolean()) {
                Object t = c1.get(i);
                c1.set(i, c2.get(i));
                c2.set(i, t);
            }
        }

        return r_.nextBoolean() ? c1 : c2;
    }

    @Override
    public String getName() {
        return "crx.vec.uniform";
    }
}
