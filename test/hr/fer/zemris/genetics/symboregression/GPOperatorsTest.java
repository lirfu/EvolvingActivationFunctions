package hr.fer.zemris.genetics.symboregression;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

public class GPOperatorsTest {

    private Random r(int i, boolean b) {
        return new Random() {
            @Override
            public int nextInt(int range) {
                return Math.min(range - 1, i);
            }

            @Override
            public boolean nextBoolean() {
                return b;
            }
        };
    }

    private TreeNodeSet buildSet() {
        TreeNodeSet set = new TreeNodeSet(r(1, true));
        set.registerTerminal(new Demo.XNode());
        set.registerBinaryOperator(new Demo.AddNode());
        return set;
    }

    private SymbolicTree buildTree() {
        return new SymbolicTree.Builder()
                .setNodeSet(new TreeNodeSet(r(1, true)))
                .add(new Demo.MulNode())
                .add(new Demo.AddNode())
                .add(new Demo.XNode())
                .add(new Demo.YNode())
                .add(new Demo.XNode())
                .build();
    }

    private String stringifyCrx(SymbolicTree p1, SymbolicTree p2, SymbolicTree c) {
        return "\np1: " + p1 + "\np2: " + p2 + "\nch: " + c;
    }

    private String stringifyMut(SymbolicTree t, SymbolicTree c) {
        return "\np: " + t + "\nc: " + c;
    }

    @Test
    public void testCrxSRSwapSubtree() {
        SymbolicTree p1 = buildTree();
        SymbolicTree p2 = buildTree();

        p2.set(4, buildTree().get(0));
        p2.set(1, buildTree().get(0));

        SymbolicTree c = new CrxSRSwapSubtree(r(1, true)).cross(p1, p2);

        assertTrue("CrxSRSwapSubtree child should differ from its parents." + stringifyCrx(p1, p2, c),
                !c.equals(p1) && !c.equals(p2));

        assertTrue("CrxSRSwapSubtree should swap subtrees correctly." + stringifyCrx(p1, p2, c),
                !c.get(1).equals(p1.get(1)) && c.get(1).equals(p2.get(1)));
    }

    @Test
    public void testMutSRSwapOrder() {
        SymbolicTree t = buildTree();
        SymbolicTree c = t.copy();

        new MutSRSwapOrder(r(1, true)).mutate(c);

        assertFalse("MutSRSwapOrder result should differ from original." + stringifyMut(t, c),
                t.equals(c));

        assertTrue("MutSRSwapOrder should mutate correctly." + stringifyMut(t, c),
                c.get(1).children_[0].equals(t.get(1).children_[1]));
    }

    @Test
    public void testMutSRReplaceSubtree() {
        SymbolicTree t = buildTree();
        SymbolicTree c = t.copy();

        new MutSRReplaceSubtree(buildSet(), new SRGenericInitializer(buildSet(), 3), r(1, true)).mutate(c);

        assertFalse("MutSRReplaceSubtree should replace the subtree." + stringifyMut(t, c),
                c.get(1).equals(t.get(1)));
    }

    @Test
    public void testMutSRReplaceNode() {
        SymbolicTree t = buildTree();
        SymbolicTree c = t.copy();

        new MutSRReplaceNode(buildSet(), r(0, true)).mutate(c);

        assertFalse("MutSRReplaceNode should replace the node." + stringifyMut(t, c),
                c.get(0).equals(t.get(0)) && c.get(1).equals(t.get(1)) && c.get(2).equals(t.get(2)));
    }

    @Test
    public void testMutSRInsertTerminal() {
        SymbolicTree t = buildTree();
        SymbolicTree c = t.copy();

        new MutSRInsertTerminal(buildSet(), r(0, true)).mutate(c);

        assertTrue("MutSRInsertTerminal should insert a terminal." + stringifyMut(t, c),
                c.get(0).getChildrenNum() == 0);
    }
}