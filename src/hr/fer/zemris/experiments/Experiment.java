package hr.fer.zemris.experiments;


public class Experiment<P> {
    private String name_;
    private final P params_;

    public Experiment(String name, P params) {
        name_ = name;
        params_ = params;
    }

    public String getName() {
        return name_;
    }

    public void setName(String name) {
        name_ = name;
    }

    public P getParams() {
        return params_;
    }
}
