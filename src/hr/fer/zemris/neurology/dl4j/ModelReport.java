package hr.fer.zemris.neurology.dl4j;

import com.google.common.base.Joiner;
import hr.fer.zemris.evolveactivationfunction.nn.CommonModel;
import hr.fer.zemris.evolveactivationfunction.nn.IModel;
import hr.fer.zemris.utils.ISerializable;
import hr.fer.zemris.utils.Pair;
import hr.fer.zemris.utils.Utilities;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.*;

/**
 * Defines the models' results from testing.
 */
public class ModelReport implements IReport, ISerializable {
    private String name_, confusion_matrix_;
    private double accuracy_, auc_ = -1., aucpr_ = -1.;
    private double avg_guess_entropy_, max_guess_entropy_, top3_accuracy_, top5_accuracy_;
    private int[] guesses_distribution_;
    private double precision_macro_, recall_macro_, f1_macro_;
    private double precision_micro_, recall_micro_, f1_micro_;
    private List<Double> train_losses_, test_losses_;

    public ModelReport() {
    }

    @Override
    public void build(String name, @NotNull IModel model, @NotNull Evaluation eval, @Nullable ROCMultiClass roc, DataSetIterator it) {
        name_ = name;
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

        confusion_matrix_ = eval.getConfusion().toCSV();
        train_losses_ = model.getTrainLosses();
        test_losses_ = model.getTestLosses();

        // Average gain entropy.
        avg_guess_entropy_ = 0;
        max_guess_entropy_ = 0;
        guesses_distribution_ = new int[it.totalOutcomes()];

        top3_accuracy_ = 0;
        top5_accuracy_ = 0;

        int inst_ctr = 0;
        it.reset();
        while (it.hasNext()) {
            DataSet ds = it.next();
            INDArray truth = ds.getLabels().argMax(1);
            INDArray pred = ((CommonModel) model).getModel().output(ds.getFeatures());

            for (int i = 0; i < pred.rows(); i++) {
                inst_ctr++;

                float[] pred_row = pred.getRow(i).toFloatVector();
                ArrayList<Pair<Float, Integer>> pred_sorted = Utilities.sortWithIndices(Utilities.objectifyArray(pred_row));

//                System.out.println(Arrays.toString(pred_row));
//                System.out.println(Arrays.toString(pred_sorted.toArray(new Pair[]{})));

                float tr = truth.getFloat(i);
                for (int s = pred_sorted.size() - 1; s >= 0; s--) {  // Iterate backwards (from max to min).
                    Pair<Float, Integer> p = pred_sorted.get(s);
                    if (p.getVal() == tr) {  // If hit.
                        int index = pred_sorted.size() - 1 - s;  // The backward index calculation (last is 0).
                        avg_guess_entropy_ += index;
                        if (max_guess_entropy_ < index) {
                            max_guess_entropy_ = index;
                        }
                        guesses_distribution_[index]++;

                        if (index < 5) {
                            top5_accuracy_++;
                            if (index < 3) {
                                top3_accuracy_++;
                            }
                        }
                        break;
                    }
                }
            }
        }
        avg_guess_entropy_ /= inst_ctr;
        top3_accuracy_ /= inst_ctr;
        top5_accuracy_ /= inst_ctr;
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

    public double avg_guess_entropy() {
        return avg_guess_entropy_;
    }

    public double max_guess_entropy() {
        return max_guess_entropy_;
    }

    public int[] guesses_distribution() {
        return guesses_distribution_;
    }

    public double top3_accuracy() {
        return top3_accuracy_;
    }

    public double top5_accuracy() {
        return top5_accuracy_;
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
                case "avg_guess_entropy":
                    avg_guess_entropy_ = Double.parseDouble(parts[1]);
                    break;
                case "max_guess_entropy":
                    max_guess_entropy_ = Double.parseDouble(parts[1]);
                    break;
                case "guesses_distribution":
                    String[] vals = parts[1].split(",");
                    guesses_distribution_ = new int[vals.length];
                    for (int i = 0; i < vals.length; i++) {
                        guesses_distribution_[i] = Integer.parseInt(vals[i]);
                    }
                    break;
                case "top3_accuracy":
                    top3_accuracy_ = Double.parseDouble(parts[1]);
                    break;
                case "top5_accuracy":
                    top5_accuracy_ = Double.parseDouble(parts[1]);
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
                .append("avg_guess_entropy").append('\t').append(avg_guess_entropy_).append('\n')
                .append("max_guess_entropy").append('\t').append(max_guess_entropy_).append('\n')
                .append("guesses_distribution").append('\t').append(Utilities.join(',', guesses_distribution_)).append('\n')
                .append("top3_accuracy").append('\t').append(top3_accuracy_).append('\n')
                .append("top5_accuracy").append('\t').append(top5_accuracy_).append('\n')
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

//    public static ModelReport averageFrom(String name, ModelReport... reports) {
//        ModelReport report = new ModelReport();
//        report.name_ = name;
//
//        report.test_losses_ = Utilities.listByRepeating(0.0, reports[0].train_losses_.size());
//        report.test_losses_ = Utilities.listByRepeating(0.0, reports[0].test_losses_.size());
//        report.confusion_matrix_ =
//
//        // Accumulate.
//        for (ModelReport mr : reports) {
//            report.accuracy_ += mr.accuracy();
//            report.precision_macro_ += mr.precision();
//            report.recall_macro_ += mr.recall();
//            report.f1_macro_ += mr.f1();
//            report.precision_micro_ += mr.precision_micro();
//            report.recall_micro_ += mr.recall_micro();
//            report.f1_micro_ += mr.f1_micro();
//            report.auc_ += mr.auc();
//            report.aucpr_ += mr.aucpr();
//
//            int i = 0;
//            for (Double v : mr.train_losses_) {
//                if (i >= report.train_losses_.size()) {
//                    report.train_losses_.add(v);
//                }
//                report.train_losses_.set(i, report.train_losses_.get(i) + v);
//                i++;
//            }
//
//            i = 0;
//            for (Double v : mr.test_losses_) {
//                if (i >= report.test_losses_.size()) {
//                    report.test_losses_.add(v);
//                }
//                report.test_losses_.set(i, report.test_losses_.get(i) + v);
//                i++;
//            }
//
//
//        }
//
//        // Average.
//        int n = reports.length;
//        report.accuracy_ /= n;
//        report.precision_macro_ /= n;
//        report.recall_macro_ /= n;
//        report.f1_macro_ /= n;
//        report.precision_micro_ /= n;
//        report.recall_micro_ /= n;
//        report.f1_micro_ /= n;
//        report.auc_ /= n;
//        report.aucpr_ /= n;
//
//        for (int i = 0; i < report.train_losses_.size(); i++) {
//            report.train_losses_.set(i, report.train_losses_.get(i) / n);
//        }
//        for (int i = 0; i < report.test_losses_.size(); i++) {
//            report.test_losses_.set(i, report.test_losses_.get(i) / n);
//        }
//
//        return report;
//    }
}
