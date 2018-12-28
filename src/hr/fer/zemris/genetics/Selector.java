package hr.fer.zemris.genetics;

public interface Selector<T extends Genotype> {
    /**
     * Selects two parents from the population. Returns the genotypes and indexes from which they were taken.
     *
     * @param population Population of Genotypes.
     * @return Array of Parent elements.
     * First two are used as parents and must define the Genotype.
     * The third is used by the elimination algorithm to know on which index to store the child (has the worst fitness).
     * If selector doesn't specify which index it needs to store results the child on, the child is <code>null</code>.
     */
    Parent[] selectParentsFrom(T[] population);

    class Parent<T extends Genotype> {
        private T genotype;
        private int index;

        public Parent(T genotype, int index) {
            this.genotype = genotype;
            this.index = index;
        }

        public T getGenotype() {
            return genotype;
        }

        public int getIndex() {
            return index;
        }
    }
}
