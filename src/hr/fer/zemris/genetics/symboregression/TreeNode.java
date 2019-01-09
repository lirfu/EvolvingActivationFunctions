package hr.fer.zemris.genetics.symboregression;


import hr.fer.zemris.utils.Counter;
import org.jetbrains.annotations.NotNull;

import java.util.LinkedList;

public abstract class TreeNode<I, O> implements IInstantiable<TreeNode<I, O>> {
    private String name_;
    protected TreeNode[] children_;
    protected Object extra_;
    private IExecutable<I, O> executable_;
    private IInstantiable<TreeNode<I, O>> instantiable_;

    protected TreeNode(String name, int children_num) {
        name_ = name;
        children_ = new TreeNode[children_num];
        executable_ = getExecutable();
        instantiable_ = getInstantiable();
    }

    protected TreeNode(String name, int children_num, Object extra) {
        this(name, children_num);
        extra_ = extra;
    }

    public TreeNode[] getChildren() {
        return children_;
    }

    public TreeNode getChild(int index) {
        return children_[index];
    }

    public int getChildrenNum() {
        return children_.length;
    }

    public String getName() {
        return name_;
    }

    public Object getExtra() {
        return extra_;
    }

    public void setExtra(Object extra) {
        extra_ = extra;
    }

    public int getDepth() {
        int max = 0; // Persume no children.
        for (TreeNode c : children_)
            if (c != null && max < c.getDepth())
                max = c.getDepth();
        return max + 1; // Return child depth + 1 for myself.
    }

    public int getSize() {
        int sum = 0; // Persume no children.
        for (TreeNode c : children_)
            if (c != null)
                sum += c.getSize();
        return sum + 1; // Return children size + 1 for myself.
    }

    protected abstract IInstantiable<TreeNode<I, O>> getInstantiable();

    @Override
    public TreeNode<I, O> getInstance() {
        return instantiable_.getInstance();
    }

    protected abstract IExecutable<I, O> getExecutable();

    public O execute(I input) {
        return executable_.execute(input, this);
    }

    /**
     * Returns the current node when counter reaches 0.
     */
    public TreeNode get(Counter ctr) {
        if (ctr.value() == 0) return this;
        for (TreeNode c : children_) {
            TreeNode r = c.get(ctr.decrement());
            if (r != null) return r;
        }
        return null;
    }

    /**
     * Swaps the contents with given node, preserves the subtrees.
     */
    public void swapNodeWith(@NotNull TreeNode n) {
        if (getChildrenNum() != n.getChildrenNum())
            throw new IllegalArgumentException("Cannot swap contents for nodes with different children number.");

        String nm = name_;
        name_ = n.name_;
        n.name_ = nm;

        IExecutable exe = executable_;
        executable_ = n.executable_;
        n.executable_ = exe;

        IInstantiable ins = instantiable_;
        instantiable_ = n.instantiable_;
        n.instantiable_ = ins;

        Object ext = extra_;
        extra_ = n.extra_;
        n.extra_ = ext;
    }

    public void swapChildrenWith(@NotNull TreeNode n) {
        if (getChildrenNum() != n.getChildrenNum())
            throw new IllegalArgumentException("Cannot swap contents for nodes with different children number.");

        TreeNode[] ch = children_;
        children_ = n.children_;
        n.children_ = ch;
    }

    /**
     * Nodes will keep their references, but swap all of their internals.
     */
    public void swapContentWith(@NotNull TreeNode n) {
        String nm = name_;
        name_ = n.name_;
        n.name_ = nm;

        IExecutable exe = executable_;
        executable_ = n.executable_;
        n.executable_ = exe;

        IInstantiable ins = instantiable_;
        instantiable_ = n.instantiable_;
        n.instantiable_ = ins;

        Object ext = extra_;
        extra_ = n.extra_;
        n.extra_ = ext;

        TreeNode[] ch = children_;
        children_ = n.children_;
        n.children_ = ch;
    }

    /**
     * Clones this node and the whole subtree under it.
     */
    @Override
    public TreeNode clone() {
        TreeNode n = getInstance();
        n.name_ = name_;
        n.instantiable_ = instantiable_;
        n.executable_ = executable_;
        n.extra_ = extra_;

        n.children_ = new TreeNode[children_.length];
        for (int i = 0; i < children_.length; i++) {
            if (children_[i] != null) {
                n.children_[i] = children_[i].clone();
            }
        }
        return n;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof TreeNode))
            return false;
        if (!name_.equals(((TreeNode) o).getName()))
            return false;
        if (children_.length != ((TreeNode) o).children_.length)
            return false;
        if (extra_ != null && !extra_.equals(((TreeNode) o).extra_))
            return false;
        for (int i = 0; i < children_.length; i++)
            if (children_[i] != null && !children_[i].equals(((TreeNode) o).children_[i]))
                return false;
        return true;
    }

    public void buildString(StringBuilder sb) {
        sb.append(extra_ == null ? name_ : extra_.toString());
        if (children_.length > 0) {
            sb.append('[');
            int i = 0;
            for (TreeNode c : children_) {
                c.buildString(sb);
                if (++i != children_.length) {
                    sb.append(',');
                }
            }
            sb.append(']');
        }
    }

    public void collect(LinkedList<TreeNode> list, Condition cond) {
        if (cond.satisfies(this)) {
            list.add(this);
        }
        for (TreeNode c : children_) {
            c.collect(list, cond);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        buildString(sb);
        return sb.toString();
    }

    public interface Condition {
        public boolean satisfies(TreeNode node);
    }
}
