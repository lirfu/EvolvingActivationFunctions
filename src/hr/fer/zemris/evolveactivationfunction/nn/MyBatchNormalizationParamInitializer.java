package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

public class MyBatchNormalizationParamInitializer extends BatchNormalizationParamInitializer {

    private static final MyBatchNormalizationParamInitializer INSTANCE = new MyBatchNormalizationParamInitializer();
    public static final String GAMMA = "gamma";
    public static final String BETA = "beta";
    public static final String GLOBAL_MEAN = "mean";
    public static final String GLOBAL_VAR = "var";

    public MyBatchNormalizationParamInitializer() {
    }

    public static MyBatchNormalizationParamInitializer getInstance() {
        return INSTANCE;
    }

    public static List<String> keys() {
        return Arrays.asList("gamma", "beta", "mean", "var");
    }

    public long numParams(NeuralNetConfiguration conf) {
        return this.numParams(conf.getLayer());
    }

    public long numParams(Layer l) {
        MyBatchNormalizationConf layer = (MyBatchNormalizationConf) l;
        return layer.isLockGammaBeta() ? 2L * layer.getNOut() : 4L * layer.getNOut();
    }

    public List<String> paramKeys(Layer layer) {
        return Arrays.asList("gamma", "beta", "mean", "var");
    }

    public List<String> weightKeys(Layer layer) {
        return Collections.emptyList();
    }

    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramView, boolean initializeParams) {
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap());
        MyBatchNormalizationConf layer = (MyBatchNormalizationConf) conf.getLayer();
        long nOut = layer.getNOut();
        long meanOffset = 0L;
        INDArray globalMeanView;
        INDArray globalVarView;
        if (!layer.isLockGammaBeta()) {
            globalMeanView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(0L, nOut)});
            globalVarView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(nOut, 2L * nOut)});
            params.put("gamma", this.createGamma(conf, globalMeanView, initializeParams));
            conf.addVariable("gamma");
            params.put("beta", this.createBeta(conf, globalVarView, initializeParams));
            conf.addVariable("beta");
            meanOffset = 2L * nOut;
        }

        globalMeanView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(meanOffset, meanOffset + nOut)});
        globalVarView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2L * nOut)});
        if (initializeParams) {
            globalMeanView.assign(Integer.valueOf(0));
            globalVarView.assign(Integer.valueOf(1));
        }

        params.put("mean", globalMeanView);
        conf.addVariable("mean");
        params.put("var", globalVarView);
        conf.addVariable("var");
        return params;
    }

    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        MyBatchNormalizationConf layer = (MyBatchNormalizationConf) conf.getLayer();
        long nOut = layer.getNOut();
        Map<String, INDArray> out = new LinkedHashMap();
        long meanOffset = 0L;
        if (!layer.isLockGammaBeta()) {
            INDArray gammaView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(0L, nOut)});
            INDArray betaView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(nOut, 2L * nOut)});
            out.put("gamma", gammaView);
            out.put("beta", betaView);
            meanOffset = 2L * nOut;
        }

        out.put("mean", gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(meanOffset, meanOffset + nOut)}));
        out.put("var", gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2L * nOut)}));
        return out;
    }

    private INDArray createBeta(NeuralNetConfiguration conf, INDArray betaView, boolean initializeParams) {
        MyBatchNormalizationConf layer = (MyBatchNormalizationConf) conf.getLayer();
        if (initializeParams) {
            betaView.assign(layer.getBeta());
        }

        return betaView;
    }

    private INDArray createGamma(NeuralNetConfiguration conf, INDArray gammaView, boolean initializeParams) {
        MyBatchNormalizationConf layer = (MyBatchNormalizationConf) conf.getLayer();
        if (initializeParams) {
            gammaView.assign(layer.getGamma());
        }

        return gammaView;
    }
}
