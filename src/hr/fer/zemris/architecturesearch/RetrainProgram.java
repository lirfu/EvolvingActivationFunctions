package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;

public class RetrainProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        String[] ds = new String[]{
                "res/noiseless_data/noiseless_all_training_256class.arff",
                "res/noiseless_data/noiseless_all_testing_256class.arff"
        };
        TreeNodeSet set = TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL);

        String experiment_name = "ReadActivations";

        String architecture = "fc(100)-fc(100)";
//        IActivation acti = new CustomFunction(new DerivableSymbolicTree(DerivableSymbolicTree.parse("relu[x]", set)));
        IActivation acti = new ActivationReLU();

        TrainParams p = StorageManager.loadTrainParameters(new Context("noiseless_all_training_256class", "common_functions_v2_fc(100)-fc(100)_relu/5_BEST"));
        TrainParams.Builder pb = new TrainParams.Builder().cloneFrom(p);

        TrainProcedureDL4J train_procedure = new TrainProcedureDL4J(ds[0], ds[1], pb);
        CommonModel model = train_procedure.createModel(new NetworkArchitecture(architecture), new IActivation[]{acti});
        Context context = train_procedure.createContext(experiment_name);

        ILogger log = new MultiLogger(new StdoutLogger(), StorageManager.createTrainingLogger(context)); // Log to stdout.
//        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
        FileStatsStorage stat_storage = null;

        log.d("===> Dataset:\n" + train_procedure.describeDatasets());
        log.d("===> Timestamp: " + LocalDateTime.now().toString());
        log.d("===> Architecture: " + architecture);
        log.d("===> Activation function: " + acti.toString());
        log.d("===> Parameters:");
        log.d(p.toString());

        Stopwatch timer = new Stopwatch();
        timer.start();
        train_procedure.train_joined(model, log, stat_storage);
        System.out.println("===> (" + Utilities.formatMiliseconds(timer.lap()) + ") Training done!");

        timer.start();
        Pair<ModelReport, Object> result = train_procedure.test(model);
        log.d("===> (" + Utilities.formatMiliseconds(timer.stop()) + ") Result:\n" + result.getKey().toString());

        StorageManager.storeTrainParameters(p, context);
        StorageManager.storeResults(result.getKey(), context);
//        StorageManager.storePredictions((INDArray) result.getVal(), context);

        train_procedure.collectModelActivationsOnTrainJoined(model, context);
    }
}
