package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class MyDenseLayerConf extends FeedForwardLayer {
    private boolean hasBias;

    private MyDenseLayerConf(MyDenseLayerConf.Builder builder) {
        super(builder);
        this.hasBias = true;
        this.hasBias = builder.hasBias;
        this.initializeConstraints(builder);
    }

    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("MyDenseLayer", this.getLayerName(), (long)layerIndex, this.getNIn(), this.getNOut());
        MyDenseLayer ret = new MyDenseLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = this.initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = this.getOutputType(-1, inputType);
        long numParams = this.initializer().numParams(this);
        int updaterStateSize = (int)this.getIUpdater().stateSize(numParams);
        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (this.getIDropout() != null) {
            trainSizeVariable = (int)((long)trainSizeVariable + inputType.arrayElementsPerExample());
        }

        trainSizeVariable = (int)((long)trainSizeVariable + outputType.arrayElementsPerExample());
        return (new org.deeplearning4j.nn.conf.memory.LayerMemoryReport.Builder(this.layerName, MyDenseLayerConf.class, inputType, outputType)).standardMemory(numParams, (long)updaterStateSize).workingMemory(0L, 0L, (long)trainSizeFixed, (long)trainSizeVariable).cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS).build();
    }

    public boolean hasBias() {
        return this.hasBias;
    }

    public boolean isHasBias() {
        return this.hasBias;
    }

    public void setHasBias(boolean hasBias) {
        this.hasBias = hasBias;
    }

    public MyDenseLayerConf() {
        this.hasBias = true;
    }

    public String toString() {
        return "DenseLayer(super=" + super.toString() + ", hasBias=" + this.isHasBias() + ")";
    }

    public boolean equals(Object o) {
        if (o == this) {
            return true;
        } else if (!(o instanceof MyDenseLayerConf)) {
            return false;
        } else {
            MyDenseLayerConf other = (MyDenseLayerConf)o;
            if (!other.canEqual(this)) {
                return false;
            } else if (!super.equals(o)) {
                return false;
            } else {
                return this.isHasBias() == other.isHasBias();
            }
        }
    }

    protected boolean canEqual(Object other) {
        return other instanceof MyDenseLayerConf;
    }

    public int hashCode() {
        int PRIME = 1;
        int result = super.hashCode();
        result = result * 59 + (this.isHasBias() ? 79 : 97);
        return result;
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.FeedForwardLayer.Builder<MyDenseLayerConf.Builder> {
        private boolean hasBias = true;

        public MyDenseLayerConf.Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        public MyDenseLayerConf build() {
            return new MyDenseLayerConf(this);
        }

        public Builder() {
        }
    }
}
