package hr.fer.zemris.data;

public class DatasetDescriptor {
    public String relation;
    public int attributes_num;
    public int classes_num;
    public int instances_num;

    public void reset() {
        relation = null;
        attributes_num = 0;
        classes_num = 0;
    }

    @Override
    public String toString() {
        return "Dataset description:\n" + relation + "\n" + attributes_num + " attributes\n" + classes_num + " classes\n" + instances_num + " instances";
    }
}
