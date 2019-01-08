package hr.fer.zemris.evolveactivationfunction;

import com.lirfu.lirfugraph.graphs.MultiLinearGraph;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Utils {
    public static MultiLinearGraph drawFunctions(double min, double max, double delta, SymbolicTree<INDArray, INDArray>... trees) {
        int size = trees.length;
        // Get names.
        String[] names = new String[size];
        for (int i = 0; i < size; i++)
            names[i] = trees[i].serialize();
        // Create the graph.
        MultiLinearGraph g = new MultiLinearGraph("Top results", size, names);
        // Calculate values and populate graph.
        for (double x = min; x <= max; x += delta) {
            double[] data = new double[size];
            for (int i = 0; i < size; i++) {
                data[i] = trees[i].execute(Nd4j.scalar(x)).getDouble(0);
            }
            g.add(data);
        }
        return g;
    }
}
