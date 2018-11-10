package hr.fer.zemris.data.modifiers;

import hr.fer.zemris.utils.Pair;

public interface IModifier {
    public void apply(Pair<float[], Float>[] data);
}
