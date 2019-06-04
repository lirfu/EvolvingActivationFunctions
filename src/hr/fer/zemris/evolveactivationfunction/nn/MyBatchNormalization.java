package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.layers.normalization.BatchNormalization;
import org.deeplearning4j.nn.layers.normalization.BatchNormalizationHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldSubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.util.OneTimeLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;


public class MyBatchNormalization extends BaseLayer<MyBatchNormalizationConf> implements IOpenLayer {
    private boolean measuring_;
    private INDArray activation_;

    public MyBatchNormalization(NeuralNetConfiguration conf) {
        super(conf);
        this.initializeHelper();
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        this.assertInputSet(false);
        INDArray s = this.preOutput(this.input, training ? TrainingMode.TRAIN : TrainingMode.TEST, workspaceMgr);
        if (measuring_)
            activation_ = s.detach();
        return s;
    }

    @Override
    public void setMeasuring(boolean measuring) {
        measuring_ = measuring;
    }

    @Override
    public INDArray getActivation() {
        return activation_;
    }

    /* Copy/paste from original ===================================================================================== */

    private static final Logger log = LoggerFactory.getLogger(BatchNormalization.class);
    BatchNormalizationHelper helper = null;
    protected int helperCountFail = 0;
    protected int index = 0;
    protected List<TrainingListener> listeners = new ArrayList();
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;


    void initializeHelper() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if ("CUDA".equalsIgnoreCase(backend)) {
            try {
                this.helper = (BatchNormalizationHelper) Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnBatchNormalizationHelper").asSubclass(BatchNormalizationHelper.class).newInstance();
                log.debug("CudnnBatchNormalizationHelper successfully initialized");
                if (!this.helper.checkSupported((this.layerConf()).getEps())) {
                    this.helper = null;
                }
            } catch (Throwable var3) {
                if (!(var3 instanceof ClassNotFoundException)) {
                    log.warn("Could not initialize CudnnBatchNormalizationHelper", var3);
                } else {
                    OneTimeLogger.info(log, "cuDNN not found: use cuDNN for better GPU performance by including the deeplearning4j-cuda module. For more information, please refer to: https://deeplearning4j.org/cudnn", new Object[]{var3});
                }
            }
        }

    }

    public double calcL2(boolean backpropParamsOnly) {
        return 0.0D;
    }

    public double calcL1(boolean backpropParamsOnly) {
        return 0.0D;
    }

    public Type type() {
        return Type.NORMALIZATION;
    }

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        this.assertInputSet(true);
        long[] shape = this.getShape(epsilon);
        long batchSize = epsilon.size(0);
        MyBatchNormalizationConf layerConf = this.layerConf();
        INDArray globalMean = (INDArray) this.params.get("mean");
        INDArray globalVar = (INDArray) this.params.get("var");
        INDArray gamma = null;
        INDArray dGlobalMeanView = (INDArray) this.gradientViews.get("mean");
        INDArray dGlobalVarView = (INDArray) this.gradientViews.get("var");
        INDArray dGammaView;
        INDArray dBetaView;
        if (layerConf.isLockGammaBeta()) {
            long[] tempShape = new long[]{1L, shape[1]};
            dGammaView = Nd4j.createUninitialized(tempShape, 'c');
            dBetaView = Nd4j.createUninitialized(tempShape, 'c');
        } else {
            gamma = this.getParam("gamma");
            dGammaView = (INDArray) this.gradientViews.get("gamma");
            dBetaView = (INDArray) this.gradientViews.get("beta");
        }

        Gradient retGradient = new DefaultGradient();
        INDArray eps;
        INDArray batchMean;
        INDArray batchVar;
        if (this.helper != null && (this.helperCountFail == 0 || !(this.layerConf()).isCudnnAllowFallback())) {
            if (layerConf.isLockGammaBeta()) {
                gamma = Nd4j.valueArrayOf(new long[]{1L, shape[1]}, layerConf.getGamma());
            }

            if (this.input.rank() == 2) {
                batchMean = this.input.reshape(this.input.ordering(), new long[]{this.input.size(0), this.input.size(1), 1L, 1L});
                eps = epsilon.reshape(epsilon.ordering(), new long[]{epsilon.size(0), epsilon.size(1), 1L, 1L});
            } else {
                batchMean = this.input;
                eps = epsilon;
            }

            Pair ret = null;

            try {
                ret = this.helper.backpropGradient(batchMean, eps, ArrayUtil.toInts(shape), gamma, dGammaView, dBetaView, layerConf.getEps(), workspaceMgr);
            } catch (Throwable var28) {
                if (!(this.layerConf()).isCudnnAllowFallback()) {
                    throw new RuntimeException("Error during MyBatchNormalization CuDNN helper backprop - isCudnnAllowFallback() is set to false", var28);
                }

                ++this.helperCountFail;
                log.warn("CuDNN MyBatchNormalization backprop execution failed - falling back on built-in implementation", var28);
            }

            if (ret != null) {
                ((Gradient) ret.getFirst()).setGradientFor("mean", dGlobalMeanView);
                ((Gradient) ret.getFirst()).setGradientFor("var", dGlobalVarView);
                if (this.input.rank() == 2) {
                    batchMean = (INDArray) ret.getSecond();
                    ret.setSecond(batchMean.reshape(batchMean.ordering(), new long[]{batchMean.size(0), batchMean.size(1)}));
                }

                batchMean = this.helper.getMeanCache();
                batchVar = this.helper.getVarCache();
                Nd4j.getExecutioner().exec(new OldSubOp(globalMean, batchMean, dGlobalMeanView));
                dGlobalMeanView.muli(1.0D - (this.layerConf()).getDecay());
                Nd4j.getExecutioner().exec(new OldSubOp(globalVar, batchVar, dGlobalVarView));
                dGlobalVarView.muli(1.0D - (this.layerConf()).getDecay());
                return ret;
            }
        }

        INDArray nextEpsilon;
        INDArray dLdVar;
        INDArray dxmu1;
        INDArray dxmu2;
        INDArray dBeta;
        if (epsilon.rank() == 2) {
            dBeta = epsilon.sum(new int[]{0});
            batchMean = epsilon.mul(this.xHat).sum(new int[]{0});
            if (layerConf.isLockGammaBeta()) {
                batchVar = epsilon.mul(layerConf.getGamma());
            } else {
                batchVar = epsilon.mulRowVector(gamma);
            }

            dLdVar = batchVar.mul(this.xMu).sum(new int[]{0}).muli(-0.5D).muli(Transforms.pow(this.std, -3.0D, true));
            dxmu1 = batchVar.sum(new int[]{0}).divi(this.std).negi();
            dxmu2 = this.xMu.sum(new int[]{0}).muli(-2.0D / (double) batchSize).muli(dLdVar);
            dxmu1 = dxmu1.addi(dxmu2);
            dxmu2 = batchVar.diviRowVector(this.std).addi(this.xMu.muliRowVector(dLdVar.muli(2.0D / (double) batchSize))).addiRowVector(dxmu1.muli(1.0D / (double) batchSize));
            dGammaView.assign(batchMean);
            dBetaView.assign(dBeta);
            retGradient.setGradientFor("gamma", dGammaView);
            retGradient.setGradientFor("beta", dBetaView);
            nextEpsilon = dxmu2;
            batchMean = this.input.mean(new int[]{0});
            eps = this.input.var(false, new int[]{0});
        } else {
            if (epsilon.rank() != 4) {
                throw new IllegalStateException("The layer prior to BatchNorm in the configuration is not currently supported. " + this.layerId());
            }

            dBeta = epsilon.sum(new int[]{0, 2, 3});
            batchMean = epsilon.mul(this.xHat).sum(new int[]{0, 2, 3});
            if (layerConf.isLockGammaBeta()) {
                batchVar = epsilon.mul(layerConf.getGamma());
            } else {
                batchVar = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(epsilon, gamma, Nd4j.createUninitialized(epsilon.shape(), epsilon.ordering()), new int[]{1}));
            }

            dLdVar = batchVar.mul(this.xMu).sum(new int[]{0, 2, 3}).muli(-0.5D).muli(Transforms.pow(this.std, -3.0D, true));
            long effectiveBatchSize = this.input.size(0) * this.input.size(2) * this.input.size(3);
            dxmu1 = batchVar.sum(new int[]{0, 2, 3}).divi(this.std).negi();
            dxmu2 = this.xMu.sum(new int[]{0, 2, 3}).muli(-2.0D / (double) effectiveBatchSize).muli(dLdVar);
            INDArray dLdmu = dxmu1.addi(dxmu2);
            INDArray dLdx = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(batchVar, this.std, batchVar, new int[]{1})).addi(Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(this.xMu, dLdVar.muli(2.0D / (double) effectiveBatchSize), this.xMu, new int[]{1})));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(dLdx, dLdmu.muli(1.0D / (double) effectiveBatchSize), dLdx, new int[]{1}));
            dGammaView.assign(batchMean);
            dBetaView.assign(dBeta);
            retGradient.setGradientFor("gamma", dGammaView);
            retGradient.setGradientFor("beta", dBetaView);
            nextEpsilon = dLdx;
            batchMean = this.input.mean(new int[]{0, 2, 3});
            eps = this.input.var(false, new int[]{0, 2, 3});
        }

        Nd4j.getExecutioner().exec(new OldSubOp(globalMean, batchMean, dGlobalMeanView));
        dGlobalMeanView.muli(1.0D - (this.layerConf()).getDecay());
        Nd4j.getExecutioner().exec(new OldSubOp(globalVar, eps, dGlobalVarView));
        dGlobalVarView.muli(1.0D - (this.layerConf()).getDecay());
        retGradient.setGradientFor("mean", dGlobalMeanView);
        retGradient.setGradientFor("var", dGlobalVarView);
        nextEpsilon = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, nextEpsilon);
        return new Pair(retGradient, nextEpsilon);
    }

    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    public Gradient gradient() {
        return this.gradient;
    }

    public INDArray preOutput(INDArray x, TrainingMode training, LayerWorkspaceMgr workspaceMgr) {
        if (x.size(1) != (this.layerConf()).getNOut()) {
            throw new IllegalArgumentException("input.size(1) does not match expected input size of " + (this.layerConf()).getNIn() + " - got input array with shape " + Arrays.toString(x.shape()));
        } else {
            MyBatchNormalizationConf layerConf = this.layerConf();
            long[] shape = this.getShape(x);
            INDArray gamma = null;
            INDArray beta = null;
            INDArray globalMeanView = this.getParam("mean");
            INDArray globalVarView = this.getParam("var");
            if (layerConf.isLockGammaBeta()) {
                if (this.helper != null && this.input.rank() == 4) {
                    long[] gammaBetaShape = new long[]{1L, (this.layerConf()).getNOut()};
                    gamma = Nd4j.valueArrayOf(gammaBetaShape, (this.layerConf()).getGamma());
                    beta = Nd4j.valueArrayOf(gammaBetaShape, (this.layerConf()).getBeta());
                }
            } else {
                gamma = this.getParam("gamma");
                beta = this.getParam("beta");
            }

            INDArray mean;
            if (this.helper != null && (this.helperCountFail == 0 || !(this.layerConf()).isCudnnAllowFallback())) {
                mean = x;
                if (x.rank() == 2) {
                    mean = x.reshape(x.ordering(), new long[]{x.size(0), x.size(1), 1L, 1L});
                }

                double decay = layerConf.getDecay();
                INDArray ret = null;

                try {
                    ret = this.helper.preOutput(mean, training == TrainingMode.TRAIN, ArrayUtil.toInts(shape), gamma, beta, globalMeanView, globalVarView, decay, layerConf.getEps(), workspaceMgr);
                } catch (Throwable var17) {
                    if (!(this.layerConf()).isCudnnAllowFallback()) {
                        throw new RuntimeException("Error during MyBatchNormalization CuDNN helper backprop - isCudnnAllowFallback() is set to false", var17);
                    }

                    ++this.helperCountFail;
                    log.warn("CuDNN MyBatchNormalization forward pass execution failed - falling back on built-in implementation", var17);
                }

                if (ret != null) {
                    if (this.input.rank() == 2) {
                        return ret.reshape(ret.ordering(), new long[]{ret.size(0), ret.size(1)});
                    }

                    return ret;
                }
            }

            INDArray var;
            if (training == TrainingMode.TRAIN) {
                switch (x.rank()) {
                    case 2:
                        mean = x.mean(new int[]{0});
                        var = x.var(false, new int[]{0});
                        break;
                    case 4:
                        mean = x.mean(new int[]{0, 2, 3});
                        var = x.var(false, new int[]{0, 2, 3});
                        break;
                    default:
                        throw new IllegalStateException("Batch normalization on activations of rank " + x.rank() + " not supported " + this.layerId());
                }

                this.std = Transforms.sqrt(workspaceMgr.dup(ArrayType.INPUT, var).addi((this.layerConf()).getEps()), false);
            } else {
                mean = this.getParam("mean");
                var = this.getParam("var");
                this.std = Transforms.sqrt(workspaceMgr.dup(ArrayType.INPUT, var).addi((this.layerConf()).getEps()), false);
            }

            INDArray activations;
            double g;
            double b;
            if (x.rank() == 2) {
                this.xMu = workspaceMgr.leverageTo(ArrayType.INPUT, x.subRowVector(mean));
                this.xHat = workspaceMgr.leverageTo(ArrayType.INPUT, this.xMu.divRowVector(this.std));
                if (layerConf.isLockGammaBeta()) {
                    g = layerConf.getGamma();
                    b = layerConf.getBeta();
                    if (g != 1.0D && b != 0.0D) {
                        activations = this.xHat.mul(g).addi(b);
                    } else {
                        activations = this.xHat;
                    }
                } else {
                    activations = this.xHat.mulRowVector(gamma).addiRowVector(beta);
                }
            } else {
                if (x.rank() != 4) {
                    throw new IllegalStateException("The layer prior to BatchNorm in the configuration is not currently supported. " + this.layerId());
                }

                if (!Shape.strideDescendingCAscendingF(x)) {
                    x = x.dup();
                }

                this.xMu = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
                this.xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, this.xMu, new int[]{1}));
                this.xHat = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
                this.xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(this.xMu, this.std, this.xHat, new int[]{1}));
                if (layerConf.isLockGammaBeta()) {
                    g = layerConf.getGamma();
                    b = layerConf.getBeta();
                    if (g != 1.0D && b != 0.0D) {
                        activations = this.xHat.mul(g).addi(b);
                    } else {
                        activations = this.xHat;
                    }
                } else {
                    activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.shape(), x.ordering());
                    activations = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(this.xHat, gamma, activations, new int[]{1}));
                    activations = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(activations, beta, activations, new int[]{1}));
                }
            }

            activations = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, activations);
            return activations;
        }
    }

    public Collection<TrainingListener> getListeners() {
        return this.listeners;
    }

    public void setListeners(TrainingListener... listeners) {
        this.listeners = new ArrayList(Arrays.asList(listeners));
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public int getIndex() {
        return this.index;
    }

    public boolean isPretrainLayer() {
        return false;
    }

    public LayerHelper getHelper() {
        return this.helper;
    }

    public long[] getShape(INDArray x) {
        if (x.rank() != 2 && x.rank() != 4) {
            if (x.rank() == 3) {
                long wDim = x.size(1);
                long hdim = x.size(2);
                if (x.size(0) > 1L && wDim * hdim == x.length()) {
                    throw new IllegalArgumentException("Illegal input for batch size " + this.layerId());
                } else {
                    return new long[]{1L, wDim * hdim};
                }
            } else {
                throw new IllegalStateException("Unable to process input of rank " + x.rank() + " " + this.layerId());
            }
        } else {
            return new long[]{1L, x.size(1)};
        }
    }

    public boolean updaterDivideByMinibatch(String paramName) {
        return !"mean".equals(paramName) && !"var".equals(paramName);
    }
}
