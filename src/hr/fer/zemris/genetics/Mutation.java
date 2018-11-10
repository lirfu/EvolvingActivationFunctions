package hr.fer.zemris.genetics;

public abstract class Mutation<G extends Genotype> extends Operator {
    public abstract void mutate(G genotype);
}
