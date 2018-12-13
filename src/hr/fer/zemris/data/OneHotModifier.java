package hr.fer.zemris.data;

import hr.fer.zemris.data.modifiers.IModifier;
import hr.fer.zemris.data.primitives.BatchPair;

import java.util.ArrayList;

public class OneHotModifier implements IModifier<BatchPair> {
    private int classes_num_;

    public OneHotModifier(int classes_num) {
        classes_num_ = classes_num;
    }

    @Override
    public void apply(ArrayList<BatchPair> data) {
        for (BatchPair bp : data) {
            for (int i = 0; i < bp.getVal().length; i++) {
                float[] hot = new float[classes_num_];
                hot[(int) bp.getVal()[i][0]] = 1;
                bp.getVal()[i] = hot;
            }
        }
    }
}
