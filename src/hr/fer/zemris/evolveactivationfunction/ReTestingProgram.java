package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.activationfunction.DerivableSymbolicTree;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Random;

public class ReTestingProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        // Set double precision globally.
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Random r = new Random(42);

        TreeNodeSet set = TreeNodeSetFactory.build(r, TreeNodeSets.ALL);

        EvolvingActivationParams.initialize(new ISerializable[]{});

        String experiment;
        String function;
        EvolvingActivationParams params;
        if (args.length == 1) {
            params = StorageManager.loadEvolutionParameters(args[0]);
            function = params.activation();
            experiment = params.experiment_name();
            if (function == null) throw new IllegalStateException("Please define a function.");
            if (params.test_path() == null) throw new IllegalStateException("Please define the test set path.");
        } else {
            function = "relu[x]";
            params = StorageManager.loadEvolutionParameters(
                    "sol/noiseless_all_training_9class/03_fixed_algorithms_train80/evolution_parameters.txt"
            );
            params.test_path("res/noiseless_Karlo/noiseless_all_testing_9class.arff");
            experiment = "00_retraining";
        }

        TrainProcedure proc = new TrainProcedure(params);
        Context c = proc.createContext(experiment);
        ILogger evo_logger = new MultiLogger(StorageManager.createEvolutionLogger(c), new StdoutLogger());

        evo_logger.i("");
        evo_logger.i(proc.describeDatasets());
        evo_logger.i("===> Re-training for function: " + function);

        SREvaluator evaluator = new SREvaluator(proc, params.architecture(), new StdoutLogger(), true);
        DerivableSymbolicTree best = new DerivableSymbolicTree(SymbolicTree.parse(function, set));
        CommonModel model = evaluator.buildModelFrom(best);

        Pair<ModelReport, INDArray> result = evaluator.evaluateModel(model, null, best.serialize());
        System.out.println(result.getKey());

        proc.storeResults(model, proc.createContext(""), result);
    }
}
