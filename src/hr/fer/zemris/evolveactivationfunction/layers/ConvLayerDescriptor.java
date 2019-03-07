package hr.fer.zemris.evolveactivationfunction.layers;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.activations.IActivation;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ConvLayerDescriptor extends ALayerDescriptor {
    private static Pattern pattern = Pattern.compile("\\((.*),(.*),(.*),(.*),(.*)\\)");
    private int kernels_num_, kernel_width_, kernel_height_;
    private int stride_x_ = 1, stride_y_ = 1;

    public ConvLayerDescriptor() {
        super("conv");
    }

    @Override
    public Layer constructLayer(IActivation activation) {
        return new ConvolutionLayer.Builder(kernel_height_, kernel_width_)
                .nOut(kernels_num_)
                .stride(stride_y_, stride_x_)
                .activation(activation)
                .convolutionMode(ConvolutionMode.Same) // Force no padding.
                .build();
    }

    @Override
    public boolean parse(String line) {
        try {
            Matcher matcher = pattern.matcher(line);
            if (matcher.find() && matcher.groupCount() >= 3) {
                kernels_num_ = Integer.parseInt(matcher.group(1));
                kernel_height_ = Integer.parseInt(matcher.group(2));
                kernel_width_ = Integer.parseInt(matcher.group(3));
                stride_y_ = Integer.parseInt(matcher.group(4));
                stride_x_ = Integer.parseInt(matcher.group(5));
                return true;
            }
        } catch (NumberFormatException e) {
        }
        return false;
    }

    @Override
    public String serialize() {
        return name_ + '(' + kernels_num_ + ',' + kernel_height_ + ',' + kernel_width_ + ',' + stride_y_ + ',' + stride_x_ + ')';
    }

    @Override
    public ALayerDescriptor newInstance() {
        return new ConvLayerDescriptor();
    }
}
