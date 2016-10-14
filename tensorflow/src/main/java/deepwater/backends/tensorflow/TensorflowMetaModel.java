package deepwater.backends.tensorflow;

public class TensorflowMetaModel {
   public static TensorflowMetaModel NullObject = new TensorflowMetaModel();
   public String train_op;
   public String save_op;
   public String inputs;
   public String summary_op;
   public String predict_op;
   public String total_loss;
   public String accuracy;
   public String labels;
   public String init;
}
