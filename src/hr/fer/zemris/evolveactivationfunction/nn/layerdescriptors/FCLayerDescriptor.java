package hr.fer.zemris.evolveactivationfunction.nn.layerdescriptors;

import hr.fer.zemris.evolveactivationfunction.nn.MyDenseLayerConf;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fully connected layer descriptor expects a layer width to be specified.
 */
public class FCLayerDescriptor extends ALayerDescriptor {
    private static Pattern pattern = Pattern.compile("\\((.*)\\)");
    private int neurons_num_;

    public FCLayerDescriptor() {
        super("fc");
    }

    @Override
    public Layer constructLayer() {
        return new MyDenseLayerConf.Builder()
                .nOut(neurons_num_)
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
