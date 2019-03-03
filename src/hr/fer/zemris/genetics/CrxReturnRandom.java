package hr.fer.zemris.genetics;


public class CrxReturnRandom<G extends Genotype> extends Crossover<G> {

    public CrxReturnRandom() {
    }

    @Override
    public String getName() {
        return "crx.return_random";
    }

    @Override
    public G cross(G parent1, G parent2) {
        return (G) (r_.nextBoolean() ? parent1.copy() : parent2.copy());
    }
}
