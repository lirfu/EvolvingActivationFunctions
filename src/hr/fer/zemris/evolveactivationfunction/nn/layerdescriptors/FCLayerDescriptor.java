package hr.fer.zemris.evolveactivationfunction.nn.layerdescriptors;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FCLayerDescriptor extends ALayerDescriptor {
    private static Pattern pattern = Pattern.compile("\\((.*)\\)");
    private int neurons_num_;

    public FCLayerDescriptor() {
        super("fc");
    }

    @Override
    public Layer constructLayer() {
        return new DenseLayer.Builder()
                .nOut(neurons_num_)
//                .activation(activation)
                .build();
    }

    @Override
    public boolean parse(String line) {
        try {
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
