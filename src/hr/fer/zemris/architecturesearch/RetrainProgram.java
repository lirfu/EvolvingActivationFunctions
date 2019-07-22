package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.Utils;
import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.*;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RetrainProgram {
    public static void main(String[] args) throws IOException, InterruptedException {
        EvolvingActivationParams.initialize(new ISerializable[]{});
        String[] ds = new String[]{
                "res/noiseless_data/noiseless_all_training_256class.arff",
                "res/noiseless_data/noiseless_all_testing_256class.arff"
        };

        // Custom params override.
        NetworkArchitecture architecture = null;
//        NetworkArchitecture architecture = new NetworkArchitecture("fc(300)-fc(300)");
        String function = null;
//        String function = "swish[elu[sin[cos[+[-[x,1.0],pow3[pow3[0.1607274601962161]]]]]]]";

        // Experiment name.
//        String orig_experiment_name = "common_functions_v2/common_functions_v2_" + architecture + "_sin[x]/6_BEST";
//        String orig_experiment_name = "gp_activation_taboo3/1";
        List<String> experiments = new ArrayList<>();
        // DPAv4 HW
//        experiments.add("gp_activation_taboo3/4");
//        experiments.add("gp_activation_taboo1/1");
//        experiments.add("gp_activation_taboo2/6");
//        experiments.add("gp_activation_taboo3/1");
//        experiments.add("gp_activation_taboo4/6");
        // DPAv4 Byte
        experiments.add("gp_activation_taboo2/1");
        experiments.add("gp_activation_taboo3/9");
        experiments.add("gp_activation_taboo0/4");
        experiments.add("gp_activation_taboo0/1");
        experiments.add("gp_activation_taboo3/1");

        for (String orig_experiment_name : experiments) {
            System.out.println("======> EXPERIMENT: " + orig_experiment_name);

            ILogger log = new StdoutLogger();
            Stopwatch timer = new Stopwatch();
            timer.start();

            Pair<CommonModel, TrainProcedureDL4J> r =
                    Utils.retrainModel(architecture, function, orig_experiment_name, ds[0], ds[1], log);
            CommonModel model = r.getKey();
            TrainProcedureDL4J train_procedure = r.getVal();
            Context c = train_procedure.createContext(orig_experiment_name);
            System.out.println("===> (" + Utilities.formatMiliseconds(timer.lap()) + ") Training done!");

            Pair<ModelReport, Object> result = train_procedure.test(model);
            StorageManager.storeCustomString(result.getKey().serialize(), c, "results_v2.txt");
            log.d("===> (" + Utilities.formatMiliseconds(timer.lap()) + ") Result:\n" + result.getKey().toString());

            //StorageManager.storeTrainParameters(p, context);
            //StorageManager.storeResults(result.getKey(), context);
            //StorageManager.storePredictions((INDArray) result.getSecond(), context);

            Triple<Double, Double, Integer> range_n_buckets = new Triple<>(-7., 7., 101);
            train_procedure.collectModelActivationsOnTest(model, c, range_n_buckets);
            System.out.println("Collection: " + Utilities.formatMiliseconds(timer.lap()));
        }
    }
}
