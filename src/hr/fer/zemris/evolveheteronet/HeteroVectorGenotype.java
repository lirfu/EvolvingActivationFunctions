package hr.fer.zemris.evolveheteronet;

import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.vector.intvector.IntVectorGenotype;

public class HeteroVectorGenotype extends IntVectorGenotype {
    private TreeNodeSet set_;

    public HeteroVectorGenotype(int size, TreeNodeSet set) {
        super(size, 0, set.getTotalSize()-1);
        set_ = set;
    }

    public HeteroVectorGenotype(HeteroVectorGenotype g) {
        super(g);
        set_ = g.set_;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < size(); i++) {
            if (i > 0)
                sb.append('-');
            sb.append(set_.getNode(get(i)));
        }
        return sb.toString();
    }
}
