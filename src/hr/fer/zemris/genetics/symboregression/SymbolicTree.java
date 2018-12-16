package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.Genotype;

import java.util.Random;
import java.util.function.Function;
import java.util.function.UnaryOperator;

public class SymbolicTree<T> extends Genotype<TreeNode<T, ?>> implements IExecutable<T, Object> {
    private int size_;
    private TreeNode root_;
    private TreeNodeSet factory_;

    public SymbolicTree(TreeNodeSet factory) {
        factory_ = factory;
    }

    private void preOrderWalk(TreeNode p, Function<TreeNode, Boolean> op) {
        if (op.apply(p)) {
            return;
        }
        if (p.getChildrenNum() > 0) {
            for (TreeNode c : p.getChildren()) {
                preOrderWalk(c, op);
            }
        }
    }

    /**
     * Execute the tree with given input.
     */
    @Override
    public Object execute(T input) {
        return root_.execute(input);
    }

    /**
     * Gets the node at given index.
     * Indexes are calculated dynamically by a pre-order preOrderWalk through the tree. Indexes start from 0.
     * <p>WARNING! Modifications to the tree can destroy previous index-node pairs.</p>
     */
    @Override
    public TreeNode get(int index) {
        TreeNode[] result = new TreeNode[1];
        preOrderWalk(root_, new Function<TreeNode, Boolean>() {
            private int i = 0;

            @Override
            public Boolean apply(TreeNode node) {
                if (i++ == index) {
                    result[0] = node;
                    return true;
                }
                return false;
            }
        });
        return result[0];
    }

    /**
     * Sets the node at given index.
     * Effectively, swaps the contents of selected and given node, preserving their references.
     * <p>Indexes are calculated dynamically by a pre-order preOrderWalk through the tree. Indexes start from 0.</p>
     * <p>WARNING! Modifications to the tree can destroy previous index-node pairs.</p>
     */
    @Override
    public void set(int index, TreeNode value) {
        preOrderWalk(root_, new Function<TreeNode, Boolean>() {
            private int i = 0;

            @Override
            public Boolean apply(TreeNode node) {
                if (i++ == index) {
                    TreeNode.swapContents(node, value);
                    return true;
                }
                return false;
            }
        });
    }

    /**
     * Number of nodes in the tree.
     */
    @Override
    public int size() {
        return size_;
    }

    public void setSize(int size) {
        size_ = size;
    }

    /**
     * Deep copy of the tree.
     */
    @Override
    public SymbolicTree<T> copy() {
        SymbolicTree<T> st = new SymbolicTree<>(factory_);
        st.root_ = (root_ == null) ? null : root_.clone(); // Clone root and its entire subtree (meaning the whole tree).
        st.size_ = size_;
        return st;
    }

    @Override
    public void initialize(Random r) {
        root_ = factory_.getRandomOperator(r);
        int[] size = new int[]{r.nextInt(3)};
        preOrderWalk(root_, n -> {
            boolean made_op = false;
            for (int i = 0; i < n.getChildrenNum(); i++) {
                if (size[0] > 0 && r.nextBoolean()) {
                    n.getChildren()[i] = factory_.getRandomOperator(r);
                    made_op = true;
                } else {
                    n.getChildren()[i] = factory_.getRandomTerminal(r);
                }
                size[0]--;
            }
            return size[0] <= 0 && !made_op;
        });
    }

    private void buildString(TreeNode p, StringBuilder sb) {
        sb.append(p.getName());
        if (p.getChildrenNum() > 0) {
            sb.append('[');
            int i = 0;
            for (TreeNode c : p.getChildren()) {
                buildString(c, sb);
                if (++i != p.getChildrenNum()) {
                    sb.append(',');
                }
            }
            sb.append(']');
        }
    }

    @Override
    public String stringify() {
        StringBuilder sb = new StringBuilder();
        buildString(root_, sb);
        return sb.toString();
    }

    @Override
    public String toString() {
        return stringify();
    }

    @Override
    public TreeNode generateParameter(Random rand) {
        return factory_.getRandomNode(rand);
    }
}
