package hr.fer.zemris.experiments;

import hr.fer.zemris.utils.Counter;
import hr.fer.zemris.utils.IBuilder;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

public class GridSearch<T> {
    private String name;

    public GridSearch(String name) {
        this.name = name;
    }

    private void mod_r(int m_i, IModifier<T>[] mods, Counter ctr, LinkedList<Experiment<T>> exs, IBuilder p) {
        if (m_i >= mods.length) {
            ctr.increment();
            exs.add(new Experiment(name + File.separator + ctr.value(), p.build()));
            return;
        }

        for (Object v : mods[m_i].getValues()) {
            mods[m_i].modify(p, v);
            mod_r(m_i + 1, mods, ctr, exs, p);
        }
    }

    public List<Experiment<T>> buildGridSearchExperiments(IBuilder p, IModifier<T>[] modifiers) {
        LinkedList<Experiment<T>> exs = new LinkedList<>();
        mod_r(0, modifiers, new Counter(), exs, p);
        return exs;
    }

    public interface IModifier<T> {
        public IBuilder<T> modify(IBuilder<T> p, Object value);

        public Object[] getValues();
    }
}
