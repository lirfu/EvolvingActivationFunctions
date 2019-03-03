package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.layers.FCLayerDescriptor;
import hr.fer.zemris.evolveactivationfunction.layers.ALayerDescriptor;
import hr.fer.zemris.utils.ISerializable;

import java.util.LinkedList;

public class NetworkArchitecture implements ISerializable {
    public static final String REGEX = "[^-]+([^-]*)[-[^-]+([^-]*)]*";
    private static final ALayerDescriptor[] AVAILABLE_LAYERS = new ALayerDescriptor[]{
            new FCLayerDescriptor()
    };

    private LinkedList<ALayerDescriptor> layers_ = new LinkedList<>();

    public NetworkArchitecture() {
    }

    public NetworkArchitecture(String s) {
        parse(s);
    }

    public NetworkArchitecture addLayer(ALayerDescriptor layer) {
        layers_.add(layer);
        return this;
    }

    public int layersNum() {
        return layers_.size();
    }

    public LinkedList<ALayerDescriptor> getLayers() {
        return layers_;
    }

    @Override
    public boolean parse(String line) {
        if (!line.matches(REGEX)) return false;

        layers_.clear();
        for (String s : line.split("-")) {
            for (ALayerDescriptor l : AVAILABLE_LAYERS) {
                if (s.startsWith(l.getName())) {
                    ALayerDescriptor d = l.newInstance();
                    if (!d.parse(s)) return false;
                    layers_.add(d);
                    break;
                }
            }
        }
        return layers_.size() > 0;
    }

    @Override
    public String serialize() {
        StringBuilder sb = new StringBuilder();
        int i = 0;
        for (ALayerDescriptor l : layers_) {
            if (i++ > 0) sb.append('-');
            sb.append(l.serialize());
        }
        return sb.toString();
    }
}
