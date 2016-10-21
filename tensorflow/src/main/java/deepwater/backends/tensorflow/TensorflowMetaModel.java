package deepwater.backends.tensorflow;

import java.util.Map;

public class TensorflowMetaModel {
   public static TensorflowMetaModel NullObject = new TensorflowMetaModel();
   public String train_op;
   public String save_filename;
   public String restore_op;
   public String save_op;

   public Map<String, String> inputs;
   public Map<String, String> outputs;
   public Map<String, String> parameters;
   public Map<String, String> metrics;

   public String summary_op;
   public String predict_op;

   public String init;
}
