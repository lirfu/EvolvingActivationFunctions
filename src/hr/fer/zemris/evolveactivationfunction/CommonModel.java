package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.neurology.dl4j.TrainParams;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

/**
 * Convenience wrapper for reducing temptation of unwanted modifying the model. Defines and builds the common model for all experiments.
 */
public class CommonModel {
    private MultiLayerNetwork model_;

    /**
     * Defines and builds the model.
     *
     * @param params      Parameters used in the training process.
     * @param layers      Array of sizes for the hidden layers.
     * @param activations Array of activation functions. Must define either one common activation activationfunction) or one activationfunction per layer.
     */
    public CommonModel(@NotNull TrainParams params, @NotNull int[] layers, @NotNull IActivation[] activations) {
        boolean common_act = false;
        if (layers.length > 1 && activations.length == 1) { // Single common activation.
            common_act = true;
        } else if (layers.length > 0 && activations.length == 0 || activations.length != layers.length) {
            throw new IllegalArgumentException("Activation activationfunction ill defined! Please provide one common activationfunction or one activationfunction per layer.");
        }

        NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(params.seed()) // Reproducibility of results (defined initialization).
                .cacheMode(CacheMode.DEVICE)
                .weightInit(WeightInit.XAVIER);
//        conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
//        conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);

        // Omittable parameters.
        if (params.decay_rate() != 1.) {
            conf.updater(new Adam(new StepSchedule(ScheduleType.EPOCH, params.learning_rate(), params.decay_rate(), params.decay_step())));
        } else {
            conf.updater(new Adam(params.learning_rate()));
        }
        if (params.regularization_coef() > 0.) {
            conf.l2(params.regularization_coef());
        }
        if (params.dropout_keep_prob() < 1.) {
            conf.dropOut(params.dropout_keep_prob());
        }
        // Apply common activationfunction globally if it is defined.
        if (common_act) {
            conf.activation(activations[0]);
        }

        // Define inner layers.
        NeuralNetConfiguration.ListBuilder list = conf.list();
        int index = 0;
        int last_size = params.input_size();
        for (int l : layers) {
            DenseLayer.Builder lay = new DenseLayer.Builder()
                    .nIn(last_size)
                    .nOut(l);
            if (!common_act) {
                lay.activation(activations[index]);
            }
            list.layer(index++, lay.build());
            last_size = l;
        }
        // Define output layer and loss.
        list.layer(index, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(last_size)
                .nOut(params.output_size())
                .build());

        model_ = new MultiLayerNetwork(list.build());
    }

    public CommonModel(MultiLayerNetwork network) {
        model_ = network;
    }

    private CommonModel(@NotNull CommonModel model) {
        model_ = model.model_.clone();
    }

    public MultiLayerNetwork getModel() {
        return model_;
    }

    @Override
    public CommonModel clone() {
        return new CommonModel(this);
    }
}
