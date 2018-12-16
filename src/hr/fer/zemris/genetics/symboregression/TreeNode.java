package hr.fer.zemris.genetics.symboregression;


public abstract class TreeNode<I, O> implements IExecutable<I, O>, IInstantiable<TreeNode> {
    private String name_;
    protected TreeNode[] children_;

    protected TreeNode(String name, int children_num) {
        name_ = name;
        children_ = new TreeNode[children_num];
    }

    public TreeNode[] getChildren() {
        return children_;
    }

    public int getChildrenNum() {
        return children_.length;
    }

    public String getName() {
        return name_;
    }

    public abstract O execute(I input);

    public abstract TreeNode getInstance();

    /**
     * Clones this node and recursively the whole subtree under it.
     */
    @Override
    public TreeNode clone() {
        TreeNode n = getInstance();
        n.name_ = name_;
        n.children_ = new TreeNode[children_.length];
        for (int i = 0; i < children_.length; i++) {
            if (n.children_[i] != null) {
                n.children_[i] = n.children_[i].clone();
            }
        }
        return n;
    }

    /**
     * Swaps the contents of given nodes.
     * Result is that the nodes keep their references, but swap contents.
     */
    public static void swapContents(TreeNode n1, TreeNode n2) {
        String nm = n1.name_;
        n1.name_ = n2.name_;
        n2.name_ = nm;

        TreeNode[] ch = n1.children_;
        n1.children_ = n2.children_;
        n2.children_ = ch;
    }
}
