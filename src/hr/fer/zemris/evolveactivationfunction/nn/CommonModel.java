package hr.fer.zemris.evolveactivationfunction.nn;

import hr.fer.zemris.evolveactivationfunction.nn.layerdescriptors.ALayerDescriptor;
import hr.fer.zemris.neurology.dl4j.TrainParams;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

/**
 * Convenience wrapper for reducing temptation of unwanted modifying the model. Defines and builds the common model for all experiments.
 */
public class CommonModel implements IModel {
    private MultiLayerNetwork model_;

    /**
     * Defines and builds the model.
     *
     * @param params       Parameters used in the training process.
     * @param architecture Architecture of the network.
     * @param activations  Array of activation functions. Must define either one common activation function) or one function per layer.
     */
    public CommonModel(@NotNull TrainParams params, @NotNull NetworkArchitecture architecture, @NotNull IActivation[] activations) {
        boolean common_act = false;
        if (architecture.layersNum() > 1 && activations.length == 1) { // Single common activation.
            common_act = true;
        } else if (architecture.layersNum() > 0 && activations.length == 0 || activations.length != architecture.layersNum()) {
            throw new IllegalArgumentException("Activation function ill defined! Please provide one common function or one function per layer.");
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
        // Apply common function globally if it is defined.
        if (common_act) {
            conf.activation(activations[0]);
        }

        // Define inner layer descriptors.
        NeuralNetConfiguration.ListBuilder list = conf.list();
        int a_i = 0, l_i = 0;
        list.setInputType(InputType.feedForward(params.input_size()));
        for (ALayerDescriptor l : architecture.getLayers()) {
            list.layer(l_i++, l.constructLayer(common_act ? activations[0] : activations[a_i++]));
            if (params.batch_norm()) {
                list.layer(l_i++, new BatchNormalization());
            }
        }
        // Define output layer and loss.
        list.layer(l_i, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
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

    @Override
    public void setModel(MultiLayerNetwork m) {
        model_ = m;
    }
}
