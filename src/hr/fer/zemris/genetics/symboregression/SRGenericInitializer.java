package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.Initializer;

public class SRGenericInitializer implements Initializer<SymbolicTree> {
    private TreeNodeSet set_;
    private Integer max_depth_;

    public SRGenericInitializer(TreeNodeSet set, int max_depth) {
        set_ = set;
        max_depth_ = max_depth;
    }

    @Override
    public void initialize(SymbolicTree genotype) {
        TreeNode root = set_.getRandomNode();
        init(root, 1);
        genotype.set(0, root);
    }

    private void init(TreeNode node, int depth) {
        for (int i = 0; i < node.getChildrenNum(); i++) {
            // When max depth is reached, generate only terminals.
            if (depth + 1 == max_depth_) {
                node.getChildren()[i] = set_.getRandomTerminal();
                continue;
            }

            // Recurse deeper.
            node.getChildren()[i] = set_.getRandomNode();
            init(node.getChildren()[i], depth + 1);
        }
    }
}
