package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.Utils;
import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Triple;
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
import org.nd4j.linalg.activations.impl.ActivationReLU6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import scala.Int;

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

        String architecture = "fc(300)-fc(300)";
        String function = "swish[elu[sin[cos[+[-[x,1.0],pow3[pow3[0.1607274601962161]]]]]]]";
//        String orig_experiment_name = "common_functions_v2/common_functions_v2_" + architecture + "_sin[x]/6_BEST";
        String orig_experiment_name = "gp_activation_taboo3/1";

        ILogger log = new StdoutLogger();
        Stopwatch timer = new Stopwatch();

        Pair<CommonModel, TrainProcedureDL4J> r = Utils.retrainModel(architecture, function, orig_experiment_name, ds[0], ds[1], log);
        CommonModel model = r.getKey();
        TrainProcedureDL4J train_procedure = r.getVal();
        System.out.println("===> (" + Utilities.formatMiliseconds(timer.lap()) + ") Training done!");

        timer.start();
        Pair<ModelReport, Object> result = train_procedure.test(model);
        log.d("===> (" + Utilities.formatMiliseconds(timer.stop()) + ") Result:\n" + result.getKey().toString());

        //StorageManager.storeTrainParameters(p, context);
        //StorageManager.storeResults(result.getKey(), context);
        //StorageManager.storePredictions((INDArray) result.getSecond(), context);

        Context c = train_procedure.createContext(orig_experiment_name);
        Triple<Double, Double, Integer> range_n_buckets = new Triple<>(-7., 7., 101);

        timer.start();
        train_procedure.collectModelActivationsOnTest(model, c, range_n_buckets);
        System.out.println("Collection: " + Utilities.formatMiliseconds(timer.stop()));
    }
}
