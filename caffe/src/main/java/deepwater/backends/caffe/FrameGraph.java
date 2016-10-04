package deepwater.backends.caffe;

import java.util.Arrays;

/**
 * Converts frame data to and from a compute graph.
 */
public class FrameGraph {
  private final Frame _frame;
  private final NetParameter _graph;
  private int _batch;
  // Input index in the frame columns
  private int[] _indices;

  // Convert column data to and from a _graph node
  static abstract class ColumnNode {
    abstract float[] read(Object value);
    abstract Object read(float[] value);
  }

  // For efficiency, mapping is determined upfront by picking the right
  // function depending on each column type. Then at runtime the function
  // is simply called.
  private ColumnNode[] _functions;
  // Input dimensions
  private int[] _axes;

  public FrameGraph(Frame frame, NetParameter graph) {
    _frame = frame;
    _graph = graph;
    _batch = (int) _graph.input_shape(0).dim(0);
    _indices = new int[_graph.input_size()];
    _functions = new FrameGraph.ColumnNode[_graph.input_size()];

    // Check inputs and map to corresponding DF column
    for (int i = 0; i < _indices.length; i++) {
      if (_graph.input_shape(i).dim(0) != _batch)
        throw new RuntimeException("Inputs of different batch sizes");
      String name = _graph.input(i).getString();
      _indices[i] = Arrays.asList(_frame.names()).indexOf(name);
      if (_indices[i] < 0)
        throw new RuntimeException("Column not found: " + name);
      _functions[i] = convert(_frame.types()[i]);
    }
  }

  /**
   * Converts from H2O columns to float arrays. Only supports simple
   * types but can be overridden.
   */
  ColumnNode convert(Class type) {
    if (type == float.class) {
      return new ColumnNode() {
        @Override
        float[] read(Object value) {
          return new float[]{(Float) value};
        }

        @Override
        Object read(float[] value) {
          return value[0];
        }
      };
    }
    if (type == float[].class) {
      return new ColumnNode() {
        @Override
        float[] read(Object value) {
          return (float[]) value;
        }

        @Override
        Object read(float[] value) {
          return value;
        }
      };
    }
    throw new RuntimeException();
  }
}