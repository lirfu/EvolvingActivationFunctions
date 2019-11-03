package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.Context;
import hr.fer.zemris.evolveactivationfunction.EvolvingActivationParams;
import hr.fer.zemris.evolveactivationfunction.StorageManager;
import hr.fer.zemris.evolveactivationfunction.Utils;
import hr.fer.zemris.evolveactivationfunction.nn.*;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.ConstNode;
import hr.fer.zemris.genetics.CrxReturnRandom;
import hr.fer.zemris.genetics.MutInitialize;
import hr.fer.zemris.genetics.symboregression.TreeNode;
import hr.fer.zemris.genetics.symboregression.TreeNodeSet;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRMeanConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapConstants;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapNodes;
import hr.fer.zemris.genetics.symboregression.crx.CrxSRSwapSubtrees;
import hr.fer.zemris.genetics.symboregression.mut.*;
import hr.fer.zemris.neurology.dl4j.ModelReport;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.*;
import hr.fer.zemris.utils.logs.*;
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

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class RetrainProgram {
    private static String[][] datasets = new String[][]{
            new String[]{
                    "res/AES_Shivam/AES_Shivam_traces_train_byte.csv ; res/AES_Shivam/AES_Shivam_labels_train_byte.csv",
                    "res/AES_Shivam/AES_Shivam_traces_test_byte.csv  ; res/AES_Shivam/AES_Shivam_labels_test_byte.csv"
            },
            new String[]{
                    "res/ASCAD/ASCAD_traces_train_byte.csv ; res/ASCAD/ASCAD_labels_train_byte.csv",
                    "res/ASCAD/ASCAD_traces_test_byte.csv  ; res/ASCAD/ASCAD_labels_test_byte.csv"
            },
            new String[]{
                    "res/Random_delay/Random_delay_traces_train_byte.csv ; res/Random_delay/Random_delay_labels_train_byte.csv",
                    "res/Random_delay/Random_delay_traces_test_byte.csv  ; res/Random_delay/Random_delay_labels_test_byte.csv"
            },
            //                new String[]{
//                        "res/DPAv2/DPAv2_traces_train_byte.csv ; res/DPAv2/DPAv2_labels_train_byte.csv",
//                        "res/DPAv2/DPAv2_traces_test_byte.csv  ; res/DPAv2/DPAv2_labels_test_byte.csv"
//                },
            new String[]{
                    "res/DPAv4/DPAv4_traces_train_byte.csv ; res/DPAv4/DPAv4_labels_train_byte.csv",
                    "res/DPAv4/DPAv4_traces_test_byte.csv  ; res/DPAv4/DPAv4_labels_test_byte.csv"
            },
    };
    static String f_root = "/home/lirfu/Desktop/DATA/sol/";
    static String[] folders = new String[]{
            "AES_Shivam_labels_train_byte/gp_age",
            "ASCAD_labels_train_byte/gp_age",
            "Random_delay_labels_train_byte/gp_age",
            "DPAv4_labels_train_byte/gp_age"
    };

    public static void main(String[] args) throws IOException, InterruptedException {
        ILogger log = new StdoutLogger();
        ILogger file_log = new FileLogger("./Results_in_short.tsv", false);
        file_log.i("exp_name\tdataset_src (trained on)\tdataset_dest (tested on)\taccuracy\taccuracy_top3\taccuracy_top5\tprecision\trecall\tf1\tavg_guess_entropy\tmax_guess_entropy");

        int i = 0, entry_index = 0;
        for (String dir_s : folders) {
            File dir = new File(f_root + dir_s);
            log.i("==========> Dataset: " + dir.getName());

            File[] fs = dir.listFiles();
            Arrays.sort(fs);
            for (File d : fs) {
                log.i("-----> Experiment: " + d.getName() + "  (index: " + entry_index++ + ")");

                String params_s = d + "/train_parameters.txt";
                // If params not given, create template and exit.
                if (!new File(params_s).exists()) { // Generate the example params.
                    log.e(">>> Can't read file: " + params_s);
                    break;
                }
                // Overwrite train/test paths.
                EvolvingActivationParams.initialize(new ISerializable[]{
                        new CrxReturnRandom()
                });
                EvolvingActivationParams params = StorageManager.loadEvolutionParameters(params_s);
                params.train_path(datasets[i][0]);
                params.test_path(datasets[i][1]);

                // Nodeset.
                Random rand = new Random(params.seed());
                TreeNodeSet set = new TreeNodeSet(new Random(42)) {
                    @Override
                    public TreeNode getNode(String node_name) {
                        TreeNode node = super.getNode(node_name);
                        if (node == null) {
                            try { // Constant node.
                                Double val = Double.parseDouble(node_name);
                                node = new ConstNode();
                                node.setExtra(val);
                            } catch (NumberFormatException ignore) {
                            }
                        }
                        return node;
                    }
                };
                set.load(TreeNodeSetFactory.build(rand, params.node_set()));

                TrainProcedureDL4J train_proc = new TrainProcedureDL4J(params);

                // Build activations
                DerivableSymbolicTree best = new DerivableSymbolicTree(DerivableSymbolicTree.parse(params.activation(), set));
                IActivation[] activations = new IActivation[params.architecture().layersNum()];
                for (int j = 0; j < activations.length; j++) {
                    activations[j] = new CustomFunction(best.copy());
                }

                // Retrain.
                log.i("===> Training network...");
                Stopwatch timer = new Stopwatch();
                timer.start();
                CommonModel model = train_proc.createModel(params.architecture(), activations);
                // Train for best amount of epochs and test.
                train_proc.train_joined(model, new DevNullLogger(), null); // Train on joined train-val set.
                log.i("(" + Utilities.formatMiliseconds(timer.stop()) + ") Done!");

                log.i("===> Collecting results on other datasets");
                for (String[] ds : datasets) {
                    ModelReport rep = train_proc.test_on(model, ds[1]).getKey();
                    file_log.i(String.join("\t", new String[]{
                            dir.getName() + '/' + d.getName(),
                            StorageManager.dsNameFromPath(datasets[i][0], false),
                            StorageManager.dsNameFromPath(ds[1], false),
                            "" + rep.accuracy(),
                            "" + rep.top3_accuracy(),
                            "" + rep.top5_accuracy(),
                            "" + rep.precision(),
                            "" + rep.recall(),
                            "" + rep.f1(),
                            "" + rep.avg_guess_entropy(),
                            "" + rep.max_guess_entropy()
                    }));
                    StorageManager.storeCustomString(rep.serialize(),
                            d + File.separator + "Fixed_results_" + StorageManager.dsNameFromPath(ds[1], false) + ".txt"
                    );
                }
                log.i("(" + Utilities.formatMiliseconds(timer.stop()) + ") Done!");
            }

            i++;
        }
    }
}
