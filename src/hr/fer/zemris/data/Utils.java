package hr.fer.zemris.data;

public class Utils {
    /**
     * Transforms the given input value to one-hot representation.
     * Assumes inputs start from 0.
     *
     * @param class_index Class value to encode (classes start from 0).
     * @param classes_num Number of possible classes (size of vector).
     * @return Vector containing a single 1 at the index of the given input class.
     */
    public static float[] toOneHot(int class_index, int classes_num) {
        return toOneHot(class_index, classes_num, 0);
    }

    /**
     * Transforms the given input value to one-hot representation.
     *
     * @param class_index Class value to encode (classes start from <code>starts_from</code>).
     * @param classes_num Number of possible classes (size of vector).
     * @param starts_from The index of the lowest class value (actual index will be <code>class_index - starts_from</code>).
     * @return Vector containing a single 1 at the index of the given input class.
     */
    public static float[] toOneHot(int class_index, int classes_num, int starts_from) {
        class_index -= starts_from;
        float[] val = new float[classes_num];
        val[class_index] = 1;
        return val;
    }

    /**
     * Transforms the one-hot class vector to the class index (starting from 0).
     *
     * @param one_hot_class One-hot vector.
     * @return Index of the firs occurrence of 1 in the given vector or <code>-1</code> if there is none.
     */
    public static int toClassIndex(float[] one_hot_class) {
        for (int i = 0; i < one_hot_class.length; i++)
            if (one_hot_class[i] == 1)
                return i;
        return -1;
    }
}
