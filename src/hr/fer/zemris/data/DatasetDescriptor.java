package hr.fer.zemris.data;

public class DatasetDescriptor {
    private String name_;
    private int attributes_num_;
    private int classes_num_;
    private int instances_num_;

    public DatasetDescriptor(String name, int attributes_num, int classes_num, int instances_num) {
        name_ = name;
        attributes_num_ = attributes_num;
        classes_num_ = classes_num;
        instances_num_ = instances_num;
    }

    public DatasetDescriptor(UnsafeDatasetDescriptor d) {
        name_ = d.name;
        attributes_num_ = d.attributes_num;
        classes_num_ = d.classes_num;
        instances_num_ = d.instances_num;
    }


    public String getName() {
        return name_;
    }

    public int getAttributesNumber() {
        return attributes_num_;
    }

    public int getClassesNumber() {
        return classes_num_;
    }

    public int getInstancesNumber() {
        return instances_num_;
    }
}
