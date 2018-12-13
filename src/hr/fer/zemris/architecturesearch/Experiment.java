package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.neurology.dl4j.TrainParams;

public class Experiment {
    private final String name_;
    private final TrainParams.Builder params_;
    private final int[] architecture_;

    public Experiment(String name, int[] architecture, TrainParams.Builder params) {
        name_ = name;
        params_ = params;
        architecture_ = architecture;
    }

    public String getName() {
        return name_;
    }

    public TrainParams.Builder getParams() {
        return params_;
    }

    public int[] getArchitecture() {
        return architecture_;
    }
}
