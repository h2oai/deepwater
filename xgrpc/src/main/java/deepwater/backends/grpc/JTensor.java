package deepwater.backends.grpc;

import java.util.Iterator;

public class JTensor<T> implements Iterable<T> {
    private final String name;

    @Override
    public Iterator<T> iterator() {
        return new Iterator<T>() {

            protected int index = 0;

            @Override
            public boolean hasNext() {
                return index < memory.length;
            }

            @Override
            public T next() {
                return memory[index++];
            }
        };
    }

    class Shape {
        Dimension[] dimensions;
    }

    class Dimension {
        int size;
    }

    protected T[] memory;
    protected int[] dimensions;

    private Shape shape;

    JTensor(String name, T[] initialValue, int[] shape) {
        this.name = name;
        memory = initialValue;
        this.dimensions = shape;
    }
}

