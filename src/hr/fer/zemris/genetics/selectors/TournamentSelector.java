package hr.fer.zemris.genetics.selectors;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.genetics.Selector;
import hr.fer.zemris.utils.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;

public class TournamentSelector implements Selector<Genotype> {
    private final int tournament_size_;
    private final Random rand_;

    public TournamentSelector(int tournament_size, Random random) {
        this.tournament_size_ = tournament_size;
        this.rand_ = random;
    }

    @SuppressWarnings("Duplicates")
    @Override
    public Parent[] selectParentsFrom(Genotype[] population) {
        // Construct indices set (guarantees uniqueness).
        ArrayList<Integer> indices = new ArrayList<>(population.length);
        for (int i = 0; i < population.length; i++) {
            indices.add(i);
        }

        // Select k random units and indices without repeats.
        ArrayList<Parent> tournament = new ArrayList<>(tournament_size_);
        for (int i = 0; i < tournament_size_; i++) {
            int index = indices.remove(rand_.nextInt(indices.size()));
            tournament.add(new Parent(population[index], index));
        }

        // Sort tournament.
        tournament.sort(Comparator.comparing(Parent::getGenotype));

        // Take first two (best) and replace the two last (worst).
        return new Parent[]{tournament.get(0), tournament.get(1), tournament.get(tournament_size_ - 1)};
    }
}
