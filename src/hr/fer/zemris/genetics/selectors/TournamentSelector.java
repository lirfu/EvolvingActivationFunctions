package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.genetics.Utils;
import hr.fer.zemris.utils.Util;

import java.util.ArrayList;
import java.util.Random;

public class TournamentSelector implements Selector {
    private int mTournamentSize;

    public TournamentSelector(int tournamentSize) {
        this.mTournamentSize = tournamentSize;
    }

    @SuppressWarnings("Duplicates")
    @Override
    public Parent[] selectParentsFrom(Random rand, Genotype[] population) {
        Genotype[] pool = new Genotype[mTournamentSize];
        ArrayList<Integer> indexes = new ArrayList<>(mTournamentSize);
        for (int i = 0; i < mTournamentSize; i++) {
            int index;
            do {
                index = rand.nextInt(population.length);
            } while (indexes.contains(index));
            pool[i] = population[index];
            indexes.add(index);
        }

        // Sort tournament.
        Utils.sortPopulation(pool);

        // Take first two best and replace the worst.
        return new Parent[]{new Parent(pool[0], indexes.get(0)), new Parent(pool[1], indexes.get(1)), new Parent(pool[pool.length - 1], indexes.get(indexes.size() - 1))};
    }
}
