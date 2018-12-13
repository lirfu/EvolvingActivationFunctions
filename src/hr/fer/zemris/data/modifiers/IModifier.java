package hr.fer.zemris.data.modifiers;


import java.util.ArrayList;

public interface IModifier<T> {
    public void apply(ArrayList<T> data);
}
