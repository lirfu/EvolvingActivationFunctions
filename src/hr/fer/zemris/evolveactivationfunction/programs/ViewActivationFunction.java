package hr.fer.zemris.evolveactivationfunction.programs;

import com.lirfu.lirfugraph.Row;
import com.lirfu.lirfugraph.VerticalContainer;
import com.lirfu.lirfugraph.Window;
import com.lirfu.lirfugraph.components.EmptySpace;
import com.lirfu.lirfugraph.components.Label;
import com.lirfu.lirfugraph.graphs.MultiLinearGraph;
import com.lirfu.lirfugraph.themes.LightTheme;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.LinkedList;
import java.util.Random;

public class ViewActivationFunction {
    public static void main(String[] args) {
        if (args.length < 1) throw new IllegalArgumentException("Specify at least 1 function.");
        TreeNodeSet set = new TreeNodeSetFactory().build(new Random(), TreeNodeSets.ALL);
        LinkedList<DerivableSymbolicTree> trees = new LinkedList<>();
        for (String s : args) {
            trees.add(new DerivableSymbolicTree(DerivableSymbolicTree.parse(s, set)));
        }

        StringBuilder legend = new StringBuilder();
        int i = 0;
        for (DerivableSymbolicTree t : trees)
            legend.append("\nf").append(++i).append(": ").append(t.serialize());

        new Window(new VerticalContainer(
                new Row(drawFunctions(-5, 5, 0.1, trees.toArray(new DerivableSymbolicTree[]{}))),
                new Row(new EmptySpace(), new Label("Legend", legend.toString()))
        ), true, true);
    }

    public static BufferedImage[] displayResult(DerivableSymbolicTree best, DerivableSymbolicTree[] top) {
        StringBuilder legend = new StringBuilder();
        legend.append("\nBest: ").append(best.serialize()).append("  (").append(best.getFitness()).append(')');
        int i = 0;
        for (SymbolicTree t : top)
            legend.append("\nf").append(++i).append(": ").append(t.serialize()).append("  (").append(t.getFitness()).append(')');

        // Display.
        MultiLinearGraph g_best = drawFunctions(-3, 3, 0.001, best);
        MultiLinearGraph g_top = drawFunctions(-3, 3, 0.001, top);

//        new Window(new VerticalContainer(
//                new Row(g_best),
//                new Row(g_top),
//                new Row(new EmptySpace(), new Label("Legend", legend.toString()))
//        ), true, true);

        return new BufferedImage[]{g_best.getImage(new Dimension(2000, 1200)), g_top.getImage(new Dimension(2000, 1200))};
    }

    public static MultiLinearGraph drawFunctions(double min, double max, double delta, DerivableSymbolicTree... trees) {
        if (trees.length < 1) throw new IllegalArgumentException("Specify at least 1 tree to draw.");
        int size = trees.length;
        // Create names.
        String[] names = new String[2 * size];
        for (int i = 0; i < size; i++) {
            names[2 * i] = "f" + (i + 1);
            names[2 * i + 1] = "d" + (i + 1);
        }
        // Create the graph.
        MultiLinearGraph g = new MultiLinearGraph(
                (trees.length > 1 ? "Top " + trees.length + " results" : "Top result"), 2 * size, names);
        g.setMinX(min).setMaxX(max).setShowDots(false);
        g.setTheme(new LightTheme());
        // Calculate values and populate graph.
        for (double x = min; x <= max; x += delta) {
            double[] data = new double[2 * size];
            for (int i = 0; i < size; i++) {
                data[2 * i] = trees[i].execute(Nd4j.scalar(x)).getDouble(0);
                data[2 * i + 1] = trees[i].derivate(Nd4j.scalar(x)).getDouble(0);
            }
            g.add(data);
        }
        return g;
    }
}
