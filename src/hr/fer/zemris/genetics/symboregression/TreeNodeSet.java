package hr.fer.zemris.genetics.symboregression;

import java.util.ArrayList;
import java.util.Random;

public class TreeNodeSet {
    private final ArrayList<TreeNode> terminals_ = new ArrayList<>();
    private final ArrayList<TreeNode> operators_ = new ArrayList<>();

    /**
     * Adds the terminal node to available nodes.
     *
     * @return <code>false</code> if the node was already added.
     */
    public boolean registerTerminalNode(TreeNode node) {
        if (terminals_.contains(node)) {
            return false;
        }
        terminals_.add(node);
        return true;
    }

    /**
     * Adds the operator node to available nodes.
     *
     * @return <code>false</code> if the node was already added.
     */
    public boolean registerOperatorNode(TreeNode node) {
        if (operators_.contains(node)) {
            return false;
        }
        operators_.add(node);
        return true;
    }

    public TreeNode getRandomNode(Random r) {
        return r.nextBoolean() ?
                terminals_.get(r.nextInt(terminals_.size())).clone() :
                operators_.get(r.nextInt(operators_.size())).clone();
    }

    public TreeNode getRandomTerminal(Random r) {
        return terminals_.get(r.nextInt(terminals_.size())).clone();
    }

    public TreeNode getRandomOperator(Random r) {
        return operators_.get(r.nextInt(operators_.size())).clone();
    }
}
