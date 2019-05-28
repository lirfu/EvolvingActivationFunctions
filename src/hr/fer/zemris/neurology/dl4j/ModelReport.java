package hr.fer.zemris.neurology.dl4j;

import com.google.common.base.Joiner;
import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Utilities;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;

import java.util.LinkedList;
import java.util.List;

/**
 * Defines the models' results from testing.
 */
public class ModelReport implements IReport, ISerializable {
    private String name_, confusion_matrix_;
    private double accuracy_, auc_, aucpr_;
    private double precision_macro_, recall_macro_, f1_macro_;
    private double precision_micro_, recall_micro_, f1_micro_;
    private List<Double> train_losses_, test_losses_;

    public ModelReport() {
    }

    @Override
    public void build(TrainParams params, IModel model, Evaluation eval, ROCMultiClass roc) {
        name_ = params.name();
        accuracy_ = eval.accuracy();

        precision_macro_ = eval.precision(EvaluationAveraging.Macro);
        recall_macro_ = eval.recall(EvaluationAveraging.Macro);
        f1_macro_ = eval.f1(EvaluationAveraging.Macro);

        precision_micro_ = eval.precision(EvaluationAveraging.Micro);
        recall_micro_ = eval.recall(EvaluationAveraging.Micro);
        f1_micro_ = eval.f1(EvaluationAveraging.Micro);

        if (roc != null) {
            auc_ = roc.calculateAverageAUC();
            aucpr_ = roc.calculateAverageAUCPR();
        }

        confusion_matrix_ = eval.confusionMatrix();
        train_losses_ = model.getTrainLosses();
        test_losses_ = model.getTestLosses();
    }

    public String name() {
        return name_;
    }

    public String confusion_matrix() {
        return confusion_matrix_;
    }

    public double accuracy() {
        return accuracy_;
    }

    public double precision() {
        return precision_macro_;
    }

    public double recall() {
        return recall_macro_;
    }

    public double f1() {
        return f1_macro_;
    }

    public double precision_micro() {
        return precision_micro_;
    }

    public double recall_micro() {
        return recall_micro_;
    }

    public double f1_micro() {
        return f1_micro_;
    }

    public double auc() {
        return auc_;
    }

    public double aucpr() {
        return aucpr_;
    }

    public List train_losses() {
        return train_losses_;
    }

    public List test_losses() {
        return test_losses_;
    }

    /**
     * @param s String containing all of the parameter lines.
     */
    @Override
    public boolean parse(String s) {
        StringBuilder cm_sb = null;
        for (String line : s.split("\n")) {
            if (cm_sb != null) {
                cm_sb.append(line).append('\n');
                continue;
            }

            String[] parts = line.split(Utilities.KEY_VALUE_SIMPLE_REGEX);
            switch (parts[0]) {
                case "name":
                    name_ = parts[1];
                    break;
                case "accuracy":
                    accuracy_ = Double.parseDouble(parts[1]);
                    break;
                case "precision":
                    precision_macro_ = Double.parseDouble(parts[1]);
                    break;
                case "recall":
                    recall_macro_ = Double.parseDouble(parts[1]);
                    break;
                case "f1":
                    f1_macro_ = Double.parseDouble(parts[1]);
                    break;
                case "precision_micro":
                    precision_micro_ = Double.parseDouble(parts[1]);
                    break;
                case "recall_micro":
                    recall_micro_ = Double.parseDouble(parts[1]);
                    break;
                case "f1_micro":
                    f1_micro_ = Double.parseDouble(parts[1]);
                    break;
                case "auc":
                    auc_ = Double.parseDouble(parts[1]);
                    break;
                case "auc_pr":
                    aucpr_ = Double.parseDouble(parts[1]);
                    break;
                case "train_losses":
                    train_losses_ = new LinkedList<>();
                    for (String v : parts[1].split(",")) {
                        train_losses_.add(Double.parseDouble(v));
                    }
                    break;
                case "test_losses":
                    test_losses_ = new LinkedList<>();
                    for (String v : parts[1].split(",")) {
                        test_losses_.add(Double.parseDouble(v));
                    }
                    break;
                case "confusion_matrix":
                    cm_sb = new StringBuilder();
                    break;
            }
        }
        if (cm_sb != null)
            confusion_matrix_ = cm_sb.toString();
        return true;
    }

    private String serialize_internal(boolean cfm) {
        StringBuilder sb = new StringBuilder()
                .append("name").append('\t').append(name_).append('\n')
                .append("accuracy").append('\t').append(accuracy_).append('\n')
                .append("precision").append('\t').append(precision_macro_).append('\n')
                .append("recall").append('\t').append(recall_macro_).append('\n')
                .append("f1").append('\t').append(f1_macro_).append('\n')
                .append("precision_micro").append('\t').append(precision_micro_).append('\n')
                .append("recall_micro").append('\t').append(recall_micro_).append('\n')
                .append("f1_micro").append('\t').append(f1_micro_).append('\n')
                .append("auc").append('\t').append(auc_).append('\n')
                .append("auc_pr").append('\t').append(aucpr_).append('\n')
                .append("train_losses").append('\t').append(train_losses_ == null ? null :
                        Joiner.on(',').join(train_losses_)).append('\n')
                .append("test_losses").append('\t').append(test_losses_ == null ? null :
                        Joiner.on(',').join(test_losses_)).append('\n');
        if (cfm)
            sb.append("confusion_matrix").append('\n').append(confusion_matrix_).append('\n');
        return sb.toString();
    }

    @Override
    public String serialize() {
        return serialize_internal(true);
    }

    @Override
    public String toString() {
        return serialize_internal(false);
    }
}
