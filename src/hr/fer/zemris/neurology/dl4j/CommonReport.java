package hr.fer.zemris.neurology.dl4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.evaluation.classification.ROCMultiClass;


public class CommonReport implements IReport {
    private String name_, confusion_matrix_;
    private double train_loss_, accuracy_, precision_, recall_, f1_, auc_, aucpr_;

    @Override
    public void build(ModelParams params, MultiLayerNetwork network, Evaluation eval, ROCMultiClass roc) {
        name_ = params.name();
        train_loss_ = network.score();
        accuracy_ = eval.accuracy();
        precision_ = eval.precision();
        recall_ = eval.recall();
        f1_ = eval.f1();
        confusion_matrix_ = eval.confusionMatrix();
        auc_ = roc.calculateAverageAUC();
        aucpr_ = roc.calculateAverageAUCPR();
    }

    public String name() {
        return name_;
    }

    public String confusion_matrix() {
        return confusion_matrix_;
    }

    public double train_loss() {
        return train_loss_;
    }

    public double accuracy() {
        return accuracy_;
    }

    public double precision() {
        return precision_;
    }

    public double recall() {
        return recall_;
    }

    public double f1() {
        return f1_;
    }

    public double auc() {
        return auc_;
    }

    public double aucpr() {
        return aucpr_;
    }

    @Override
    public String toString() {
        return new StringBuilder()
                .append("name: ").append(name_).append('\n')
                .append("train_loss: ").append(train_loss_).append('\n')
                .append("accuracy: ").append(accuracy_).append('\n')
                .append("precision: ").append(precision_).append('\n')
                .append("recall: ").append(recall_).append('\n')
                .append("f1: ").append(f1_).append('\n')
                .append("auc: ").append(auc_).append('\n')
                .append("auc_pr: ").append(aucpr_).append('\n')
                .append("confusion_matrix:\n").append(confusion_matrix_)
                .toString();
    }
}
