package hr.fer.zemris.genetics;

import java.util.Random;

public interface Selector {
    /**
     * Selects two parents from the population. Returns the genotypes and indexes from which they were taken.
     *
     * @param rand       Random generator.
     * @param population Population of Genotypes.
     * @return Array of Parent elements.
     * First two are used as parents and must define the Genotype.
     * The third is used by the elimination algorithm to know on which index to store the child (has the worst Genotype). If selector doesn't specify which index it needs to store the child on, the child is <code>null</code>.
     */
    public Parent[] selectParentsFrom(Random rand, Genotype[] population);

    public static class Parent {
        private Genotype genotype;
        private int index;

        public Parent(Genotype genotype, int index) {
            this.genotype = genotype;
            this.index = index;
        }

        public Genotype getGenotype() {
            return genotype;
        }

        public int getIndex() {
            return index;
        }
    }
}
