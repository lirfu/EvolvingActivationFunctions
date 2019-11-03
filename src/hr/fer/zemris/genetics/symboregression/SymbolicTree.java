package hr.fer.zemris.genetics.symboregression;

import hr.fer.zemris.genetics.Genotype;
import hr.fer.zemris.utils.Counter;

import java.util.LinkedList;
import java.util.Random;

public class SymbolicTree<I, O> extends Genotype<TreeNode<I, O>> {
    private TreeNodeSet set_;
    private int size_;
    protected TreeNode<I, O> root_;

    public SymbolicTree(TreeNodeSet set) {
        set_ = set;
    }

    public SymbolicTree(TreeNodeSet set, TreeNode<I, O> root) {
        set_ = set;
        root_ = root;
        updateSize();
    }

    public SymbolicTree(SymbolicTree<I, O> t) {
        super(t);
        set_ = t.set_;
        root_ = t.root_ == null ? null : t.root_.clone();
        updateSize();
    }

    /**
     * Execute the tree with given input.
     */
    public O execute(I input) {
        return root_.execute(input);
    }

    /**
     * Gets the node at given index.
     * Indexes are calculated dynamically by a pre-order preOrderWalk through the tree. Indexes start from 0.
     * <p>WARNING! Modifications to the tree can destroy previous index-node relations.</p>
     */
    @Override
    public TreeNode<I, O> get(int index) {
        if (index == 0) {
            return root_;
        } else if (index > size_) {
            throw new IllegalStateException("Index out of bounds: " + index);
        }
        return root_.get(new Counter(index));
    }

    /**
     * Sets the node at given index.
     * Effectively, swaps the contents of selected and given node, preserving their references.
     * <p>Indexes are calculated dynamically by a pre-order preOrderWalk through the tree. Indexes start from 0.</p>
     * <p>If index is 0, the given value is set as the tree root. This effectively re-defines the whole tree.</p>
     * <p>WARNING! Method swaps the contents of the given node with the node at position. This means replaced node will be in the given value.</p>
     * <p>WARNING! Modifications to the tree can destroy previous index-node relations.</p>
     */
    @Override
    public void set(int index, TreeNode<I, O> value) {
        if (index > size_) {
            throw new IllegalStateException("Index out of bounds: " + index);
        }
        if (index == 0 && root_ == null) {
            root_ = value;
        } else {
            get(index).swapAllWith(value);
        }
        updateSize();
    }

    /**
     * Number of nodes in the tree.
     */
    @Override
    public int size() {
        return size_;
    }

    public int depth() {
        return root_ != null ? root_.getDepth() : 0;
    }

    public void updateSize() {
        size_ = (root_ == null) ? 0 : root_.getSize();
    }

    public void collect(TreeNode.Condition c, LinkedList<TreeNode> list) {
        root_.collect(list, c);
    }

    @Override
    public SymbolicTree<I, O> generateInstance(Random rand) {
        return new SymbolicTree<>(set_, set_.getRandomNode());
    }

    /**
     * Deep copy of the tree.
     */
    @Override
    public SymbolicTree<I, O> copy() {
        SymbolicTree t = new SymbolicTree<>(set_, (root_ == null) ? null : root_.clone());
        t.fitness_ = fitness_;
        return t;
    }

    @Override
    public String serialize() {
        if (root_ == null)
            return "null";
        return root_.toString();
    }


    @Override
    public String toString() {
        return serialize();
    }

    @Override
    public boolean parse(String str) {
        Builder b = new Builder().setNodeSet(set_);

        if (str.indexOf('[') < 0) { // Root is terminal.
            b.add(set_.getNode(str));
        } else {
            String[] parts = str.split("[\\[,\\]]+");
            for (String s : parts) {
                b.add(set_.getNode(s));
            }
        }
        root_ = b.root_;
        return true;
    }

    public static SymbolicTree parse(String str, TreeNodeSet set) {
        SymbolicTree tree = new SymbolicTree(set);
        tree.parse(str);
        return tree;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof SymbolicTree)) return false;
        if (size_ != ((SymbolicTree) o).size_) return false;
        return root_.equals(((SymbolicTree) o).root_);
    }

    /**
     * Convenience class for building the symbolic tree.
     */
    public static class Builder {
        protected TreeNodeSet set_;
        protected TreeNode root_ = null;
        protected boolean is_full_ = false;

        public SymbolicTree build() {
            if (set_ == null)
                throw new IllegalStateException("Node set must be defined!");

            add(null);  // Simulate adding to set the "is_full" flag if tree is complete. This won't be added to the full tree.
            if (!is_full_)
                throw new IllegalStateException("The tree isn't complete (some children are undefined)! " + new SymbolicTree<>(set_, root_).toString());

            return new SymbolicTree(set_, root_);
        }

        private boolean add(TreeNode current, TreeNode value) {
            for (int i = 0; i < current.getChildrenNum(); i++) {
                if (current.getChildren()[i] == null) {
                    current.getChildren()[i] = value;
                    return true;
                }
                if (add(current.getChildren()[i], value))
                    return true;
            }
            return false;
        }

        /**
         * Puts the given node to the first encountered null child in an in-order walk.
         * If there is no space left, the inputs are ignored.
         */
        public Builder add(TreeNode node) {
            if (root_ == null) {
                root_ = node;
            } else if (!add(root_, node)) {
                is_full_ = true;
            }
            return this;
        }

        public Builder setNodeSet(TreeNodeSet set) {
            set_ = set;
            return this;
        }
    }
}
