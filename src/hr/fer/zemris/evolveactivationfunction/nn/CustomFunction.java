package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import javassist.bytecode.SignatureAttribute;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

public class CustomFunction extends BaseActivationFunction implements Serializable {
    static {
        NeuralNetConfiguration.registerLegacyCustomClassesForJSON(CustomFunction.class);
    }

    private transient DerivableSymbolicTree tree_; // Don't object serialize this structure.
    private String serialized;

    public CustomFunction() {
    }

    public CustomFunction(DerivableSymbolicTree tree) {
        tree_ = tree;
        serialized = tree.serialize();
    }

    /**
     * Initialize the tree from serialized string. Used after object deserialization.
     */
    public void initialize() {
        tree_ = new DerivableSymbolicTree(DerivableSymbolicTree.parse(serialized, TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL)));
    }

    @Override
    public INDArray getActivation(INDArray input, boolean b) {
        if (tree_ == null) { // Because DL4J uses some hidden magic to serialize objects to JSON, I'm forced to put first-time initialization here.
            initialize();
        }
        try (MemoryWorkspace ws = input.isAttached() ?
                input.data().getParentWorkspace().notifyScopeBorrowed() :
                Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            return tree_.execute(input);
        }
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray input, INDArray epsilon) {
        try (MemoryWorkspace ws = input.isAttached() ?
                input.data().getParentWorkspace().notifyScopeBorrowed() :
                Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            INDArray dLds = tree_.derivate(input);
            dLds.muli(epsilon);
            return new Pair<>(dLds, epsilon);
        }
    }

    /**
     * Override default Java serialization. Serialize function into a string, then serialize the object.
     */
    private void writeObject(ObjectOutputStream out) throws IOException {
        serialized = tree_.serialize();
        out.defaultWriteObject();
    }

    /**
     * Override default Java deserialization. Deserialize object and function from string and build the tree.
     */
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        initialize();
    }

    @Override
    public String toString() {
        return tree_.serialize();
    }
}
