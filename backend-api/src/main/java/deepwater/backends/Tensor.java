package deepwater.backends;

public interface Tensor {
    enum TensorType {
        DT_FLOAT,
        DT_DOUBLE
    }

    TensorType getType();
    String getName();

    float[] getFloatArray();
    double[] getDoubleArray();
}
