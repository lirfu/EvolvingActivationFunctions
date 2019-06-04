package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

import java.util.*;

public class MyBatchNormalizationConf extends FeedForwardLayer {
    protected double decay;
    protected double eps;
    protected boolean isMinibatch;
    protected double gamma;
    protected double beta;
    protected boolean lockGammaBeta;
    protected boolean cudnnAllowFallback;

    private MyBatchNormalizationConf(MyBatchNormalizationConf.Builder builder) {
        super(builder);
        this.decay = 0.9D;
        this.eps = 1.0E-5D;
        this.isMinibatch = true;
        this.gamma = 1.0D;
        this.beta = 0.0D;
        this.lockGammaBeta = false;
        this.cudnnAllowFallback = true;
        this.decay = builder.decay;
        this.eps = builder.eps;
        this.isMinibatch = builder.isMinibatch;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
        this.lockGammaBeta = builder.lockGammaBeta;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
        this.initializeConstraints(builder);
    }

    public MyBatchNormalizationConf() {
        this(new MyBatchNormalizationConf.Builder());
    }

    public MyBatchNormalizationConf clone() {
        MyBatchNormalizationConf clone = (MyBatchNormalizationConf)super.clone();
        return clone;
    }

    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNOutSet("MyBatchNormalizationConf", this.getLayerName(), (long)layerIndex, this.getNOut());
        MyBatchNormalization ret = new MyBatchNormalization(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = this.initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    public ParamInitializer initializer() {
        return MyBatchNormalizationParamInitializer.getInstance();
    }

    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input type: Batch norm layer expected input of type CNN, got null for layer \"" + this.getLayerName() + "\"");
        } else {
            switch(inputType.getType()) {
                case FF:
                case CNN:
                case CNNFlat:
                    return inputType;
                default:
                    throw new IllegalStateException("Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got " + inputType + " for layer index " + layerIndex + ", layer name = " + this.getLayerName());
            }
        }
    }

    public void setNIn(InputType inputType, boolean override) {
        if (this.nIn <= 0L || override) {
            switch(inputType.getType()) {
                case FF:
                    this.nIn = ((InputType.InputTypeFeedForward)inputType).getSize();
                    break;
                case CNN:
                    this.nIn = ((InputType.InputTypeConvolutional)inputType).getChannels();
                    break;
                case CNNFlat:
                    this.nIn = ((InputType.InputTypeConvolutionalFlat)inputType).getDepth();
                default:
                    throw new IllegalStateException("Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got " + inputType + " for layer " + this.getLayerName() + "\"");
            }

            this.nOut = this.nIn;
        }

    }

    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType.getType() == InputType.Type.CNNFlat) {
            InputType.InputTypeConvolutionalFlat i = (InputType.InputTypeConvolutionalFlat)inputType;
            return new FeedForwardToCnnPreProcessor(i.getHeight(), i.getWidth(), i.getDepth());
        } else {
            return inputType.getType() == InputType.Type.RNN ? new RnnToFeedForwardPreProcessor() : null;
        }
    }

    public double getL1ByParam(String paramName) {
        return 0.0D;
    }

    public double getL2ByParam(String paramName) {
        return 0.0D;
    }

    public IUpdater getUpdaterByParam(String paramName) {
        byte var3 = -1;
        switch(paramName.hashCode()) {
            case 116519:
                if (paramName.equals("var")) {
                    var3 = 3;
                }
                break;
            case 3020272:
                if (paramName.equals("beta")) {
                    var3 = 0;
                }
                break;
            case 3347397:
                if (paramName.equals("mean")) {
                    var3 = 2;
                }
                break;
            case 98120615:
                if (paramName.equals("gamma")) {
                    var3 = 1;
                }
        }

        switch(var3) {
            case 0:
            case 1:
                return this.iUpdater;
            case 2:
            case 3:
                return new NoOp();
            default:
                throw new IllegalArgumentException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = this.getOutputType(-1, inputType);
        long numParams = this.initializer().numParams(this);
        int updaterStateSize = 0;

        String s;
        for(Iterator var6 = BatchNormalizationParamInitializer.keys().iterator(); var6.hasNext(); updaterStateSize = (int)((long)updaterStateSize + this.getUpdaterByParam(s).stateSize(this.nOut))) {
            s = (String)var6.next();
        }

        long inferenceWorkingSize = 2L * inputType.arrayElementsPerExample();
        long trainWorkFixed = 2L * this.nOut;
        long trainWorkingSizePerExample = inferenceWorkingSize + outputType.arrayElementsPerExample() + 2L * this.nOut;
        return (new org.deeplearning4j.nn.conf.memory.LayerMemoryReport.Builder(this.layerName, MyBatchNormalizationConf.class, inputType, outputType)).standardMemory(numParams, (long)updaterStateSize).workingMemory(0L, 0L, trainWorkFixed, trainWorkingSizePerExample).cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS).build();
    }

    public boolean isPretrainParam(String paramName) {
        return false;
    }

    public double getDecay() {
        return this.decay;
    }

    public double getEps() {
        return this.eps;
    }

    public boolean isMinibatch() {
        return this.isMinibatch;
    }

    public double getGamma() {
        return this.gamma;
    }

    public double getBeta() {
        return this.beta;
    }

    public boolean isLockGammaBeta() {
        return this.lockGammaBeta;
    }

    public boolean isCudnnAllowFallback() {
        return this.cudnnAllowFallback;
    }

    public void setDecay(double decay) {
        this.decay = decay;
    }

    public void setEps(double eps) {
        this.eps = eps;
    }

    public void setMinibatch(boolean isMinibatch) {
        this.isMinibatch = isMinibatch;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public void setLockGammaBeta(boolean lockGammaBeta) {
        this.lockGammaBeta = lockGammaBeta;
    }

    public void setCudnnAllowFallback(boolean cudnnAllowFallback) {
        this.cudnnAllowFallback = cudnnAllowFallback;
    }

    public String toString() {
        return "MyBatchNormalizationConf(super=" + super.toString() + ", decay=" + this.getDecay() + ", eps=" + this.getEps() + ", isMinibatch=" + this.isMinibatch() + ", gamma=" + this.getGamma() + ", beta=" + this.getBeta() + ", lockGammaBeta=" + this.isLockGammaBeta() + ", cudnnAllowFallback=" + this.isCudnnAllowFallback() + ")";
    }

    public boolean equals(Object o) {
        if (o == this) {
            return true;
        } else if (!(o instanceof MyBatchNormalizationConf)) {
            return false;
        } else {
            MyBatchNormalizationConf other = (MyBatchNormalizationConf)o;
            if (!other.canEqual(this)) {
                return false;
            } else if (!super.equals(o)) {
                return false;
            } else if (Double.compare(this.getDecay(), other.getDecay()) != 0) {
                return false;
            } else if (Double.compare(this.getEps(), other.getEps()) != 0) {
                return false;
            } else if (this.isMinibatch() != other.isMinibatch()) {
                return false;
            } else if (Double.compare(this.getGamma(), other.getGamma()) != 0) {
                return false;
            } else if (Double.compare(this.getBeta(), other.getBeta()) != 0) {
                return false;
            } else if (this.isLockGammaBeta() != other.isLockGammaBeta()) {
                return false;
            } else {
                return this.isCudnnAllowFallback() == other.isCudnnAllowFallback();
            }
        }
    }

    protected boolean canEqual(Object other) {
        return other instanceof MyBatchNormalizationConf;
    }

    public int hashCode() {
        int PRIME = 1;
        int result = super.hashCode();
        long $decay = Double.doubleToLongBits(this.getDecay());
        result = result * 59 + (int)($decay >>> 32 ^ $decay);
        long $eps = Double.doubleToLongBits(this.getEps());
        result = result * 59 + (int)($eps >>> 32 ^ $eps);
        result = result * 59 + (this.isMinibatch() ? 79 : 97);
        long $gamma = Double.doubleToLongBits(this.getGamma());
        result = result * 59 + (int)($gamma >>> 32 ^ $gamma);
        long $beta = Double.doubleToLongBits(this.getBeta());
        result = result * 59 + (int)($beta >>> 32 ^ $beta);
        result = result * 59 + (this.isLockGammaBeta() ? 79 : 97);
        result = result * 59 + (this.isCudnnAllowFallback() ? 79 : 97);
        return result;
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.FeedForwardLayer.Builder<MyBatchNormalizationConf.Builder> {
        protected double decay = 0.9D;
        protected double eps = 1.0E-5D;
        protected boolean isMinibatch = true;
        protected boolean lockGammaBeta = false;
        protected double gamma = 1.0D;
        protected double beta = 0.0D;
        protected List<LayerConstraint> betaConstraints;
        protected List<LayerConstraint> gammaConstraints;
        protected boolean cudnnAllowFallback = true;

        public Builder(double decay, boolean isMinibatch) {
            this.decay = decay;
            this.isMinibatch = isMinibatch;
        }

        public Builder(double gamma, double beta) {
            this.gamma = gamma;
            this.beta = beta;
        }

        public Builder(double gamma, double beta, boolean lockGammaBeta) {
            this.gamma = gamma;
            this.beta = beta;
            this.lockGammaBeta = lockGammaBeta;
        }

        public Builder(boolean lockGammaBeta) {
            this.lockGammaBeta = lockGammaBeta;
        }

        public Builder() {
        }

        public MyBatchNormalizationConf.Builder minibatch(boolean minibatch) {
            this.isMinibatch = minibatch;
            return this;
        }

        public MyBatchNormalizationConf.Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public MyBatchNormalizationConf.Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        public MyBatchNormalizationConf.Builder eps(double eps) {
            this.eps = eps;
            return this;
        }

        public MyBatchNormalizationConf.Builder decay(double decay) {
            this.decay = decay;
            return this;
        }

        public MyBatchNormalizationConf.Builder lockGammaBeta(boolean lockGammaBeta) {
            this.lockGammaBeta = lockGammaBeta;
            return this;
        }

        public MyBatchNormalizationConf.Builder constrainBeta(LayerConstraint... constraints) {
            this.betaConstraints = Arrays.asList(constraints);
            return this;
        }

        public MyBatchNormalizationConf.Builder constrainGamma(LayerConstraint... constraints) {
            this.gammaConstraints = Arrays.asList(constraints);
            return this;
        }

        public MyBatchNormalizationConf.Builder cudnnAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return this;
        }

        public MyBatchNormalizationConf build() {
            return new MyBatchNormalizationConf(this);
        }

        public Builder(double decay, double eps, boolean isMinibatch, boolean lockGammaBeta, double gamma, double beta, List<LayerConstraint> betaConstraints, List<LayerConstraint> gammaConstraints, boolean cudnnAllowFallback) {
            this.decay = decay;
            this.eps = eps;
            this.isMinibatch = isMinibatch;
            this.lockGammaBeta = lockGammaBeta;
            this.gamma = gamma;
            this.beta = beta;
            this.betaConstraints = betaConstraints;
            this.gammaConstraints = gammaConstraints;
            this.cudnnAllowFallback = cudnnAllowFallback;
        }
    }
}
