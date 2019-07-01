package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.nn.CustomFunction;
import hr.fer.zemris.evolveactivationfunction.nn.NetworkArchitecture;
import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedureDL4J;
import hr.fer.zemris.evolveactivationfunction.tree.DerivableSymbolicTree;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSetFactory;
import hr.fer.zemris.evolveactivationfunction.tree.TreeNodeSets;
import hr.fer.zemris.evolveactivationfunction.tree.nodes.*;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Stopwatch;
import hr.fer.zemris.utils.Utilities;
import hr.fer.zemris.utils.logs.ILogger;
import hr.fer.zemris.utils.logs.MultiLogger;
import hr.fer.zemris.utils.logs.StdoutLogger;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.IActivation;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Random;

public class Utils {
    public static String toLatex(SymbolicTree t) {
        String token = "<->";
        final String[] str = new String[]{token};
        t.collect(n -> {
            String s;
            switch (n.getName()) {
                case AddNode.NAME:
                    s = "(" + token + " + " + token + ")";
                    break;
                case SubNode.NAME:
                    s = "(" + token + " - " + token + ")";
                    break;
                case MulNode.NAME:
                    s = token + " \\\\cdot " + token;
                    break;
                case DivNode.NAME:
                    s = "\\\\frac{" + token + "}{" + token + "}";
                    break;
                case MinNode.NAME:
                    s = "\\\\min (" + token + "," + token + ")";
                    break;
                case MaxNode.NAME:
                    s = "\\\\max (" + token + "," + token + ")";
                    break;
                case SinNode.NAME:
                    s = "\\\\sin (" + token + ")";
                    break;
                case CosNode.NAME:
                    s = "\\\\cos (" + token + ")";
                    break;
                case TanNode.NAME:
                    s = "\\\\tan (" + token + ")";
                    break;
                case ExpNode.NAME:
                    s = "\\\\exp (" + token + ")";
                    break;
                case Pow2Node.NAME:
                    s = "(" + token + ")^2";
                    break;
                case Pow3Node.NAME:
                    s = "(" + token + ")^3";
                    break;
                case PowNode.NAME:
                    s = "(" + token + ")^{" + token + "}";
                    break;
                case LogNode.NAME:
                    s = "\\\\log (" + token + ")";
                    break;
                case InputNode.NAME:
                    s = "x";
                    break;
                case ConstNode.NAME:
                    s = String.valueOf((Double) n.getExtra());
                    break;
                case ReLUNode.NAME:
                    s = "ReLU (" + token + ")";
                    break;
                case SigmoidNode.NAME:
                    s = "\\\\sigma (" + token + ")";
                    break;
                case GaussNode.NAME:
                    s = "gauss (" + token + ")";
                    break;
                default:
                    s = n.getName() + " (" + token + ")";
            }
            str[0] = str[0].replaceFirst(token, s);
            return false;
        }, null);
        return str[0];
    }

    public static Pair<CommonModel, TrainProcedureDL4J> retrainModel(String arch, String function, String experiment_name, String train_ds_path, String test_ds_path, ILogger log) throws IOException, InterruptedException {
        IActivation acti = new CustomFunction(new DerivableSymbolicTree(
                DerivableSymbolicTree.parse(function, TreeNodeSetFactory.build(new Random(), TreeNodeSets.ALL))
        ));
        TrainParams p = StorageManager.loadTrainParameters(new Context(
                StorageManager.dsNameFromPath(train_ds_path, false),
                experiment_name)
        );
        TrainParams.Builder pb = new TrainParams.Builder().cloneFrom(p);

        TrainProcedureDL4J train_procedure = new TrainProcedureDL4J(train_ds_path, test_ds_path, pb);
        CommonModel model = train_procedure.createModel(new NetworkArchitecture(arch), new IActivation[]{acti});

//        FileStatsStorage stat_storage = StorageManager.createStatsLogger(context);
        FileStatsStorage stat_storage = null;

        log.d("===> Dataset:\n" + train_procedure.describeDatasets());
        log.d("===> Timestamp: " + LocalDateTime.now().toString());
        log.d("===> Architecture: " + arch);
        log.d("===> Activation function: " + acti.toString());
        log.d("===> Parameters:");
        log.d(p.toString());

        train_procedure.train_joined(model, log, stat_storage);

        return new Pair<>(model, train_procedure);
    }
}
