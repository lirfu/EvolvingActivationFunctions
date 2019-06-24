package hr.fer.zemris.genetics.vector.mut;

import hr.fer.zemris.genetics.Mutation;
import hr.fer.zemris.genetics.vector.AVectorGenotype;

public class MutVGenerateSingle extends Mutation<AVectorGenotype> {

    @Override
    public void mutate(AVectorGenotype g) {
        g.set(r_.nextInt(g.size()), g.generateParameter(r_));
    }

    @Override
    public String getName() {
        return "mut.vec.generate_single";
    }
}
