package hr.fer.zemris.genetics;

public abstract class Crossover<G extends Genotype> extends Operator<Crossover> {
    public abstract G cross(G parent1, G parent2);
}
