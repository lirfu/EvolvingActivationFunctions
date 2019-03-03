package hr.fer.zemris.evolveactivationfunction.layers;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FCLayerDescriptor extends ALayerDescriptor {
    private int neurons_num_;

    public FCLayerDescriptor() {
        super("fc");
    }

    public String getName() {
        return name_;
    }

    @Override
    public int outputNum() {
        return neurons_num_;
    }

    @Override
    public Layer constructLayer(int input_num, IActivation activation) {
        return new DenseLayer.Builder()
                .nIn(input_num)
                .nOut(neurons_num_)
                .activation(activation)
                .build();
    }

    @Override
    public boolean parse(String line) {
        try {
            Pattern pattern = Pattern.compile("\\((.*)\\)");
            Matcher matcher = pattern.matcher(line);
            if (matcher.find()) {
                neurons_num_ = Integer.parseInt(matcher.group(1));
                return true;
            }
        } catch (NumberFormatException e) {
        }
        return false;
    }

    @Override
    public String serialize() {
        return name_ + '(' + neurons_num_ + ')';
    }

    @Override
    public ALayerDescriptor newInstance() {
        return new FCLayerDescriptor();
    }
}
