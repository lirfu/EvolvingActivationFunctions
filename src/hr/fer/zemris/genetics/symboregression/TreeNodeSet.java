package hr.fer.zemris.genetics.symboregression;

import org.jetbrains.annotations.NotNull;
import sun.reflect.generics.tree.Tree;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class TreeNodeSet {
    private Random r_;
    private ArrayList<ArrayList<TreeNode>> node_buckets_;

    public TreeNodeSet(Random r) {
        r_ = r;
        node_buckets_ = new ArrayList<>();
        node_buckets_.add(new ArrayList<>());
        node_buckets_.add(new ArrayList<>());
        node_buckets_.add(new ArrayList<>());
    }

    public TreeNodeSet(TreeNodeSet set) {
        r_ = set.r_;
        node_buckets_ = set.node_buckets_;
    }

    public void load(TreeNodeSet set) {
        r_ = set.r_;
        node_buckets_ = set.node_buckets_;
    }

    @Override
    public String toString() {
        String separator = ", ";
        StringBuilder sb = new StringBuilder();
        for (ArrayList<TreeNode> arr : node_buckets_)
            for (TreeNode n : arr)
                sb.append(n.getName()).append(separator);
        return sb.toString().substring(0, sb.length() - separator.length());
    }

    /**
     * Registers node to a bucket, based on its children number.
     */
    public boolean registerNode(TreeNode node) {
        ArrayList<TreeNode> l = node_buckets_.get(node.getChildrenNum());
        if (l.contains(node)) {
            return false;
        }
        l.add(node);
        return true;
    }

    /**
     * Adds the terminal node to available nodes.
     *
     * @return <code>false</code> if the node was already added.
     */
    public boolean registerTerminal(TreeNode node) {
        if (node.getChildrenNum() != 0)
            throw new IllegalArgumentException("Terminal operator must have no children.");
        return registerNode(node);
    }

    /**
     * Adds the unary operator node to available nodes.
     *
     * @return <code>false</code> if the node was already added.
     */
    public boolean registerUnaryOperator(TreeNode node) {
        if (node.getChildrenNum() != 1)
            throw new IllegalArgumentException("Unary operator must have exactly 1 child.");
        return registerNode(node);
    }

    /**
     * Adds the binary operator node to available nodes.
     *
     * @return <code>false</code> if the node was already added.
     */
    public boolean registerBinaryOperator(TreeNode node) {
        if (node.getChildrenNum() != 2)
            throw new IllegalArgumentException("Binary operator must have exactly 2 children.");
        return registerNode(node);
    }

    /* GETTERS */

    private TreeNode getRandomNodeFrom(ArrayList<TreeNode> l) {
        if (l.size() == 0)
            return null;
        return l.get(r_.nextInt(l.size())).clone();
    }

    public TreeNode getRandomTerminal() {
        TreeNode n = getRandomNodeFrom(node_buckets_.get(0));
        if (n == null)
            throw new IllegalStateException("No terminals registered.");
        return n;
    }

    public TreeNode getRandomUnaryOperator() {
        return getRandomNodeFrom(node_buckets_.get(1));
    }

    public TreeNode getRandomBinaryOperator() {
        return getRandomNodeFrom(node_buckets_.get(2));
    }

    public TreeNode getRandomOperator() {
        if (node_buckets_.get(1).isEmpty()) {
            return getRandomBinaryOperator();
        } else if (node_buckets_.get(2).isEmpty()) {
            return getRandomUnaryOperator();
        } else {
            return r_.nextBoolean() ? getRandomUnaryOperator() : getRandomBinaryOperator();
        }
    }

    public TreeNode getRandomNode() {
        return r_.nextBoolean() ? getRandomTerminal() : getRandomOperator();
    }

    public TreeNode getRandomNode(int children_num) {
        if (children_num > node_buckets_.size())
            return null;
        return getRandomNodeFrom(node_buckets_.get(children_num));
    }

    public TreeNode getNode(String node_name) {
        for (List<TreeNode> l : node_buckets_) {
            for (TreeNode n : l) {
                if (n.getName().equals(node_name)) {
                    return n.clone();
                }
            }
        }
        return null;
    }

    public static class Builder {
        private Random r_ = new Random();
        private LinkedList<TreeNode> nodes_ = new LinkedList<>();

        public TreeNodeSet build() {
            TreeNodeSet set = new TreeNodeSet(r_);
            for (TreeNode n : nodes_) {
                set.registerNode(n);
            }
            return set;
        }

        public Builder addNode(@NotNull TreeNode node) {
            nodes_.add(node);
            return this;
        }

        public Builder setRandom(@NotNull Random random) {
            r_ = random;
            return this;
        }
    }
}
