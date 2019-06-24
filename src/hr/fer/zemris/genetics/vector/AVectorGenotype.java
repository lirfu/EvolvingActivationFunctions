package hr.fer.zemris.genetics.vector;

import hr.fer.zemris.genetics.Genotype;

public abstract class AVectorGenotype<T> extends Genotype<T> {
    protected AVectorGenotype() {
        super();
    }

    protected AVectorGenotype(AVectorGenotype<T> g) {
        super(g);
    }
}
