package hr.fer.zemris.experiments;


public class Experiment<P> {
    private final String name_;
    private final P params_;

    public Experiment(String name, P params) {
        name_ = name;
        params_ = params;
    }

    public String getName() {
        return name_;
    }

    public P getParams() {
        return params_;
    }
}
