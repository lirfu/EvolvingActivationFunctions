package hr.fer.zemris.genetics.vector.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.vector.AVectorGenotype;

public class MutVGenerateMultiple extends Mutation<AVectorGenotype> {
    private int max_change_;

    public MutVGenerateMultiple(int max_change) {
        max_change_ = max_change;
    }

    @Override
    public void mutate(AVectorGenotype g) {
        int change = r_.nextInt(max_change_) + 1;  // Change at least 1.
        for (int i = 0; i < change; i++) {
            g.set(r_.nextInt(g.size()), g.generateParameter(r_));
        }
    }

    @Override
    public String getName() {
        return "mut.vec.generate_multiple";
    }
    @Override
    public String serialize() {
        return super.serialize()
                + serializeKeyVal(getName() + ".max_change", String.valueOf(max_change_));
    }
}
