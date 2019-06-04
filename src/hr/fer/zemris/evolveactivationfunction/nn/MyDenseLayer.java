package hr.fer.zemris.evolveactivationfunction.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;


public class MyDenseLayer extends BaseLayer<MyDenseLayerConf> implements IOpenLayer {
    private boolean measuring_;
    private INDArray activation_;

    public MyDenseLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public MyDenseLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray s = super.activate(training, workspaceMgr);
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

    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public boolean hasBias() {
        return this.layerConf().hasBias();
    }
}
