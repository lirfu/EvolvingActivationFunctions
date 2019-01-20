package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableSymbolicTree;
import hr.fer.zemris.genetics.CrxReturnRandom;
import hr.fer.zemris.genetics.MutInitialize;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRMeanConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapNodes;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapSubtrees;
import hr.fer.zemris.genetics.symboregression.mut.*;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Random;

public class ReTestingProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Random r = new Random(42);

        TreeNodeSet set = TreeNodeSetFactory.build(r, TreeNodeSetFactory.Set.ALL);

        EvolvingActivationParams.initialize(new ISerializable[]{});

        String function = "min[sin[sin[x]],x]";
        EvolvingActivationParams params = StorageManager.loadEvolutionParameters(
                "sol/noiseless_all_training_256class/06_arch28-28-28/evolution_parameters.txt"
        );
        params.test_path("res/noiseless_Karlo/noiseless_all_testing_256class.arff");

        TrainProcedure proc = new TrainProcedure(params);
        System.out.println(proc.describeDatasets());

        SREvaluator evaluator = new SREvaluator(proc, params.architecture(), new StdoutLogger(), true);
        DerivableSymbolicTree best = new DerivableSymbolicTree(SymbolicTree.parse(function, set));
        CommonModel model = evaluator.buildModelFrom(best);

        Pair<ModelReport, INDArray> result = evaluator.evaluateModel(model, null, best.serialize());
        System.out.println(result.getKey());
    }
}
