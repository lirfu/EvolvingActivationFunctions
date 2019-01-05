package hr.fer.zemris.genetics;

public class MutInitialize<G extends Genotype> extends Mutation<G> {
    private Initializer<G> initializer_;

    public MutInitialize(Initializer<G> initializer) {
        initializer_ = initializer;
    }

    @Override
    public String getName() {
        return "mut.initialize";
    }

    @Override
    public void mutate(G genotype) {
        initializer_.initialize(genotype);
    }
}
