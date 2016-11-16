package deepwater.backends.tensorflow;

import java.util.HashMap;
import java.util.Map;

public class TensorflowMetaModel {
   public static TensorflowMetaModel NullObject = new TensorflowMetaModel();
   public String train_op = "";
   public String save_filename = "";
   public String restore_op = "";
   public String save_op = "";

   public Map<String, String> inputs = new HashMap<>();
   public Map<String, String> outputs = new HashMap<>();
   public String summary_op = "";
   public String predict_op = "";
   public String init = "";
   Map<String, String> parameters = new HashMap<>();
   Map<String, String> metrics = new HashMap<>();


}
