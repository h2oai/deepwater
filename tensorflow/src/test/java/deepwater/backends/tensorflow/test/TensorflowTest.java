package deepwater.backends.tensorflow.test;


import deepwater.backends.tensorflow.models.ModelFactory;
import deepwater.backends.tensorflow.models.TensorflowModel;
import deepwater.datasets.MNISTImageDataset;
import deepwater.datasets.Pair;
import org.bytedeco.javacpp.tensorflow;
import org.junit.Ignore;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.LineNumberReader;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.bytedeco.javacpp.tensorflow.Cast;
import static org.bytedeco.javacpp.tensorflow.Const;
import static org.bytedeco.javacpp.tensorflow.ConvertGraphDefToGraph;
import static org.bytedeco.javacpp.tensorflow.DT_FLOAT;
import static org.bytedeco.javacpp.tensorflow.DT_STRING;
import static org.bytedeco.javacpp.tensorflow.DecodeJpeg;
import static org.bytedeco.javacpp.tensorflow.DecodePng;
import static org.bytedeco.javacpp.tensorflow.Div;
import static org.bytedeco.javacpp.tensorflow.DotGraph;
import static org.bytedeco.javacpp.tensorflow.DotOptions;
import static org.bytedeco.javacpp.tensorflow.Env;
import static org.bytedeco.javacpp.tensorflow.ExpandDims;
import static org.bytedeco.javacpp.tensorflow.GraphConstructorOptions;
import static org.bytedeco.javacpp.tensorflow.GraphDef;
import static org.bytedeco.javacpp.tensorflow.GraphDefBuilder;
import static org.bytedeco.javacpp.tensorflow.Node;
import static org.bytedeco.javacpp.tensorflow.OpRegistry;
import static org.bytedeco.javacpp.tensorflow.OpDef;
import static org.bytedeco.javacpp.tensorflow.ReadBinaryProto;
import static org.bytedeco.javacpp.tensorflow.ReadFile;
import static org.bytedeco.javacpp.tensorflow.Reshape;
import static org.bytedeco.javacpp.tensorflow.ResizeBilinear;
import static org.bytedeco.javacpp.tensorflow.Session;
import static org.bytedeco.javacpp.tensorflow.SessionOptions;
import static org.bytedeco.javacpp.tensorflow.Squeeze;
import static org.bytedeco.javacpp.tensorflow.Status;
import static org.bytedeco.javacpp.tensorflow.StringArray;
import static org.bytedeco.javacpp.tensorflow.StringTensorPairVector;
import static org.bytedeco.javacpp.tensorflow.StringVector;
import static org.bytedeco.javacpp.tensorflow.Sub;
import static org.bytedeco.javacpp.tensorflow.Tensor;
import static org.bytedeco.javacpp.tensorflow.TensorShape;
import static org.bytedeco.javacpp.tensorflow.TensorVector;
import static org.bytedeco.javacpp.tensorflow.TopKV2;

public class TensorflowTest {

    static String expandPath(String path) {
        return path.replaceFirst("^~", System.getProperty("user.home"));
    }

    Status loadImage(ImageParams imageParams, TensorVector outputs) throws Exception {
        String image_reader_name = "image_reader";
        String output_name = "normalize";

        Node image_reader;
        GraphDefBuilder graphBuilder = new GraphDefBuilder();
        Node file_reader = ReadFile(Const(imageParams.getImagePath(), graphBuilder.opts()), graphBuilder.opts().WithName(image_reader_name));

        if (imageParams.getImagePath().endsWith(".png")) {
            image_reader = DecodePng(file_reader, graphBuilder.opts().WithAttr("channels", 3).WithName("png_reader"));
        } else {
            image_reader = DecodeJpeg(file_reader, graphBuilder.opts().WithAttr("channels", 3).WithName("jpeg_reader"));
        }
        // cast to float
        Node float_caster = Cast(image_reader, DT_FLOAT, graphBuilder.opts().WithName("float_caster"));

        Node dims_expander = ExpandDims(float_caster, Const(0, graphBuilder.opts()), graphBuilder.opts());
        Node resized = ResizeBilinear(dims_expander, Const(new int[]{imageParams.getInput_height(), imageParams.getInput_width()}, graphBuilder.opts().WithName("size")), graphBuilder.opts());
        Node normalized = Div(
                Sub(resized, Const(imageParams.getInput_mean(), graphBuilder.opts()), graphBuilder.opts().WithName("subtraction")),
                Const(imageParams.getInput_std(), graphBuilder.opts()),
                graphBuilder.opts());
        Squeeze(normalized, graphBuilder.opts());
        Reshape(normalized, Const(new int[]{-1, 784}, graphBuilder.opts()), graphBuilder.opts().WithName(output_name));

        tensorflow.GraphDef graph = new tensorflow.GraphDef();
        Status status = graphBuilder.ToGraphDef(graph);

        checkStatus(status);

        Session session = new Session(new SessionOptions());
        status = session.Create(graph);
        checkStatus(status);

        status = session.Run(new StringTensorPairVector(),
                new StringVector(output_name), new StringVector(), outputs);
        checkStatus(status);
        return Status.OK();
    }

    TensorVector getTopKLabels(Session sess, Tensor distribution, int k) {

        final String output_name = "top_k";

        GraphDefBuilder b = new GraphDefBuilder();

        TopKV2(Const(distribution, b.opts()), Const(k, b.opts()), b.opts().WithName("top_k"));

        tensorflow.GraphDef graph = new tensorflow.GraphDef();
        Status status = b.ToGraphDef(graph);
        checkStatus(status);
        status = sess.Extend(graph);
        checkStatus(status);
        TensorVector outputs = new TensorVector();
        status = sess.Run(new StringTensorPairVector(),
                new StringVector(new String[]{output_name + ":0", output_name + ":1"}),
                new StringVector(),
                outputs);
        checkStatus(status);
        return outputs;
    }

    void checkStatus(Status status) {
        if (!status.ok()) {
            throw new InternalError(status.error_message().getString());
        }
    }


    @Test
    public void trainMNIST() throws Exception {

        SessionOptions opt = new SessionOptions();
        Session sess = new Session(opt);
        GraphDef graph_def = new GraphDef();
        InputStream stream = getClass().getResourceAsStream("mnist.pb");

        TensorflowModel model = ModelFactory.LoadModel("LENET");

        checkStatus(sess.Create(model.getGraph()));


        for (int i = 0; i < graph_def.node_size(); i++) {
            System.out.println(">>>>> " + graph_def.node(i).name().getString());
        }

        MNISTImageDataset dataset = new MNISTImageDataset("/home/fmilo/workspace/h2o-3/train-labels-idx1-ubyte.gz",
                "/home/fmilo/workspace/h2o-3/train-images-idx3-ubyte.gz");
        List<Pair<Integer, float[]>> images = dataset.loadDigitImages();

        int wrong_prediction = 0;
        int right_prediction = 0;
        int batch_size = 100;

        int current_batch_size = 0;

        initVariables(sess);

        float[] batch = new float[784 * batch_size];
        float[] batch_labels = new float[10 * batch_size];

        for (int ii = 0; ii < 5; ii++) {

            for (Pair<Integer, float[]> sample : images) {
                if (current_batch_size == 0) {
                    Arrays.fill(batch_labels, 0);
                    Arrays.fill(batch, 0);
                }
                float[] array = sample.getValue();
                // accumulate images and labels into a batch
                System.arraycopy(array, 0, batch, current_batch_size * 784, array.length);

                int pos = sample.getKey();
                batch_labels[current_batch_size * 10 + pos] = 1;
                current_batch_size++;
                if (current_batch_size < batch_size) {
                    continue;
                }

                trainDigit(sess, batch, batch_labels, batch_size);

                current_batch_size = 0;
            }

            inferMNISTSess(sess);
        }
    }

    @Test
    public void inferMNIST() throws Exception {

        SessionOptions opt = new SessionOptions();
        Session sess = new Session(opt);
        TensorflowModel model = ModelFactory.LoadModel("LENET");

        Status status = sess.Create(model.getGraph());
        if (!status.ok()) {
            throw new InternalError("could not create graph definition");
        }

        initVariables(sess);
        inferMNISTSess(sess);
    }

    void inferMNISTSess(Session sess) throws Exception {
        MNISTImageDataset dataset = new MNISTImageDataset("/home/fmilo/workspace/h2o-3/t10k-labels-idx1-ubyte.gz", "/home/fmilo/workspace/h2o-3/t10k-images-idx3-ubyte.gz");
        List<Pair<Integer, float[]>> images = dataset.loadDigitImages();

        int wrong_prediction = 0;
        int right_prediction = 0;
        int batch_size = 100;
        float[] batch = new float[784 * batch_size];
        int[] batch_labels = new int[batch_size];
        int current_batch_size = 0;
        for (Pair<Integer, float[]> sample : images) {
            if (current_batch_size == 0) {
                Arrays.fill(batch_labels, 0);
                Arrays.fill(batch, 0);
            }
            float[] array = sample.getValue();
            // accumulate images and labels into a batch
            System.arraycopy(array, 0, batch, current_batch_size * 784, array.length);
            batch_labels[current_batch_size] = sample.getKey();
            current_batch_size++;
            if (current_batch_size < batch_size) {
                continue;
            }

            int[] predictions = inferDigit(sess, batch, batch_size);
            for (int i = 0; i < predictions.length; i++) {
                if (predictions[i] != batch_labels[i]) {
                    wrong_prediction++;
                } else {
                    right_prediction++;
                }
            }
            current_batch_size = 0;
        }

        System.out.println("right predictions: " + right_prediction + " - wrong Predictions:" + wrong_prediction);
        System.out.println("accuracy:" + right_prediction / 1.0 * (right_prediction + wrong_prediction));
    }

    void initVariables(Session sess){
        TensorVector outputs = new TensorVector();
        Status status = sess.Run(new StringTensorPairVector(),
                new StringVector(),
                new StringVector("init"), outputs);
        checkStatus(status);
    }

    int[] trainDigit(Session sess, float[] data, float[] y, int batch_size) throws Exception {
        TensorVector outputs = new TensorVector();
        ImageParams params = new ImageParams(expandPath("~/workspace/mnist_png/mnist_png/testing/0/1416.png"), 28, 28, 128, 128);
        loadImage(params, outputs); // "input/x-input", "layer2/activation", outputs);

        Tensor result = outputs.get(0);

        outputs = new TensorVector();

        Tensor y_batch = new Tensor(DT_FLOAT, new TensorShape(batch_size, 10));
        ((FloatBuffer) y_batch.createBuffer()).put(y);

        Tensor mnist_batch_image = new Tensor(DT_FLOAT, new TensorShape(batch_size, 784));
        ((FloatBuffer) mnist_batch_image.createBuffer()).put(data);

        Tensor dropout = new Tensor(DT_FLOAT, new TensorShape(1));

        FloatBuffer dropout_b = dropout.createBuffer();
        dropout_b.put(1.0f);

        Status status = sess.Run(new StringTensorPairVector(new String[]{"input/x-input", "input/y-input",
                        "dropout/Placeholder"},
                        new Tensor[]{mnist_batch_image, y_batch, dropout}),
                new StringVector("accuracy/accuracy/Mean:0"),
                new StringVector("train/Adam"), outputs);
        checkStatus(status);
//        getTopKLabels(outputs.get(0), 1);
//
       FloatBuffer fb = outputs.get(0).createBuffer();
       float[] activation_layer = new float[1];
       fb.get(activation_layer);
//
//        int[] indexes = new int[batch_size];
//        ((IntBuffer) outputs.get(1).createBuffer()).get(indexes);
//
       // for (int i = 0; i < activation_layer.length; i++) {
       //     System.out.println("accuracy:" + activation_layer[i]);
       // }

        return new int[]{} ; //indexes;

    }

    int[] inferDigit(Session sess, float[] data, int batch_size) throws Exception {
        TensorVector outputs = new TensorVector();
        ImageParams params = new ImageParams(expandPath("~/workspace/mnist_png/mnist_png/testing/0/1416.png"), 28, 28, 128, 128);
        loadImage(params, outputs); // "input/x-input", "layer2/activation", outputs);

        Tensor result = outputs.get(0);

        outputs = new TensorVector();

        Tensor mnist_batch_image = new Tensor(DT_FLOAT, new TensorShape(batch_size, 784));

        ((FloatBuffer) mnist_batch_image.createBuffer()).put(data);
        Tensor dropout = new Tensor(DT_FLOAT, new TensorShape(1));

        FloatBuffer dropout_b = dropout.createBuffer();
        dropout_b.put(1.0f);

        Status status = sess.Run(new StringTensorPairVector(new String[]{"input/x-input", "dropout/Placeholder"},
                        new Tensor[]{mnist_batch_image, dropout}),
                new StringVector("layer2/activation"),
                new StringVector(), outputs);
        checkStatus(status);
        //getTopKLabels(sess, outputs.get(0), 1);

        FloatBuffer fb = outputs.get(0).createBuffer();
        float[] activation_layer = new float[10 * batch_size];
        fb.get(activation_layer);

        int[] indexes = new int[batch_size];

        for (int j = 0; j < batch_size; j++) {
            int argmax =0;
            float maxvalue = 0.0f;
            for (int i = 0; i < 10; i++) {
                float value = activation_layer[j * 10 + i];
                if (value > maxvalue){
                    argmax = i;
                    maxvalue = value;
                }
            }
            indexes[j] = argmax;
        }

        return indexes;

    }

    public Tensor asTensor(String value) {
        Tensor t = new Tensor(DT_STRING, new TensorShape(value.length()));
        t.createStringArray().put(value);
        return t;
    }

    @Ignore("needs to update the inception model")
    @Test
    public void inferInception() throws Exception {
        SessionOptions opt = new SessionOptions();
        Session sess = new Session(opt);

        GraphDef graph_def = new GraphDef();

        ArrayList<OpDef> list = new ArrayList<>();

        Status status = ReadBinaryProto(Env.Default(), expandPath("~/workspace/deepwater/tensorflow/models/inception/classify_image_graph_def.pb"), graph_def);
        checkStatus(status);
        status = sess.Create(graph_def);
        checkStatus(status);


        HashMap<Integer, String> uid_to_human_name = loadMapping(expandPath("~/workspace/deepwater/tensorflow/models/inception/imagenet_2012_challenge_label_map_proto.pbtxt"),
                expandPath("~/workspace/deepwater/tensorflow/models/inception/imagenet_synset_to_human_label_map.txt"));


        String image_path = expandPath("~/workspace/deepwater/tensorflow/models/inception/cropped_panda.jpg");
        BufferedImage bimg = ImageIO.read(new File(image_path));

        byte[] img = Files.readAllBytes(Paths.get(image_path));
        TensorVector outputs = new TensorVector();
        Tensor image_data = new Tensor(DT_STRING, new TensorShape(img.length));
        StringArray buffer_image_data = image_data.createStringArray();
        //buffer_image_data.put(new StringArray().data().put(img));
        buffer_image_data.resize(img.length);
        buffer_image_data.data().put(img);
        assert buffer_image_data.size() == img.length : image_data.dim_size(0) + " " + img.length;

        Tensor image_path_t = asTensor(new String(img));
        status = sess.Run(new StringTensorPairVector(new String[]{"DecodeJpeg/contents"}, new Tensor[]{image_data}),
                new StringVector("softmax"), new StringVector(), outputs);
        checkStatus(status);

        FloatBuffer softmax = outputs.get(0).createBuffer();
        TensorVector results = getTopKLabels(sess, outputs.get(0), 5);

        FloatBuffer topK = results.get(0).createBuffer();
        IntBuffer topKindex = results.get(1).createBuffer();
        assert topK.limit() == topKindex.limit();
        for (int i = 0; i < topK.limit(); i++) {
            System.out.println(i + " " + topK.get(i) + " " + uid_to_human_name.get(topKindex.get(i)));
        }

        GraphConstructorOptions opts = new GraphConstructorOptions();
        tensorflow.Graph gg = new tensorflow.Graph(OpRegistry.Global());
        ConvertGraphDefToGraph(opts, graph_def, gg);

        DotOptions gopts = new DotOptions();



        String s = DotGraph(gg, gopts).getString();
        //String buff = new String(dotValue.asBuffer().array(), Charset.forName("UTF-8"));

        System.out.println(s);
    }


    HashMap<Integer, String> loadMapping(String label_map_proto, String label_map) {
        Pattern regexp = Pattern.compile("^n(\\d+)(\\s+)([ \\S,]*)");
        Matcher matcher = regexp.matcher("");
        HashMap<Integer, String> uid_to_class_name = new HashMap<>();
        try {
            // Read the uid -> Class mapping
            BufferedReader buff_reader = new BufferedReader(new FileReader(label_map));
            LineNumberReader reader = new LineNumberReader(buff_reader);
            String line;
            while ((line = reader.readLine()) != null) {
                matcher.reset(line);
                if (matcher.find()) {
                    uid_to_class_name.put(Integer.parseInt(matcher.group(1)), matcher.group(3));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        HashMap<Integer, String> result = new HashMap<>();
        regexp = Pattern.compile("^.*?(\\d+).*");
        matcher = regexp.matcher("");

        try {
            // Read the uid -> int mapping
            BufferedReader buff_reader = new BufferedReader(new FileReader(label_map_proto));
            LineNumberReader reader = new LineNumberReader(buff_reader);
            List<Integer> values = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                // skip comments. This is needed or the regex will pick up the number inside it.
                if (line.startsWith("#")) {
                    continue;
                }
                matcher.reset(line);
                if (matcher.find()) {
                    values.add(Integer.parseInt(matcher.group(1)));
                }
            }
            assert values.size() % 2 == 0 : values;

            // consume the array as a sequence of [(uid,id),...]
            for (int i = 0; i < values.size(); i += 2) {
                int index = values.get(i);
                int uid = values.get(i + 1);
                // remap uid to id
                result.put(index, uid_to_class_name.get(uid));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return result;
    }

    /*
    @Test
    public void TestKerasModelInMemoryProcess() throws Exception {
        System.setProperty("jpy.debug", "true");
        System.setProperty("jpy.jdlLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jdl.so");
        System.setProperty("jpy.pythonLib", "/home/fmilo/anaconda2/envs/h2o/lib/libpython2.7.so");
        System.setProperty("jpy.jpyLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jpy.so");

        PyLib.startPython();
        InputStream in = new FileInputStream(expandPath("~/workspace/h2o-3/h2o-algos/src/test/java/hex/deepwater/inception_v3.py"));
        String code = IOUtils.toString(in, StandardCharsets.UTF_8);



        PyModule.executeCode("import os;os.environ['KERAS_BACKEND'] = 'tensorflow'", PyInputMode.STATEMENT);
        //PyObject result = PyLib.executeCode(code, new Map<String,Object>(), new Map<String,Object>());
        PyLib.execScript(code);
        PyLib.execScript("print(locals())");

        Map<String,Object> locals = new HashMap<>();
        Map<String,Object> globals = new HashMap<>();

        PyObject module = PyModule.executeCode(code, PyInputMode.SCRIPT, locals, globals);

        PyModule.getMain().getAttribute("model").getAttribute("optimizer").setAttribute("learning_rate", 1.0, Double.class );

        PyModule.executeCode("print(model.optimizer.learning_rate)", PyInputMode.SCRIPT);

    }

    @Test
    public void TestIPythonModelInMemoryProcess() throws Exception {
        System.setProperty("jpy.debug", "true");
        System.setProperty("jpy.jdlLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jdl.so");
        System.setProperty("jpy.pythonLib", "/home/fmilo/anaconda2/envs/h2o/lib/libpython2.7.so");
        System.setProperty("jpy.jpyLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jpy.so");

        PyLib.startPython();
        InputStream in = new FileInputStream(expandPath("~/workspace/h2o-3/h2o-algos/src/test/java/hex/deepwater/ipython_start.py"));
        String code = IOUtils.toString(in, StandardCharsets.UTF_8);

        //PyObject result = PyLib.executeCode(code, new Map<String,Object>(), new Map<String,Object>());
        PyLib.execScript(code);
        //PyLib.execScript("print(locals())");

        Map<String,Object> locals = new HashMap<>();
        Map<String,Object> globals = new HashMap<>();

        PyObject module = PyModule.executeCode(code, PyInputMode.SCRIPT, locals, globals);

        PyModule.getMain().getAttribute("model").getAttribute("optimizer").setAttribute("learning_rate", 1.0, Double.class );

        PyModule.executeCode("print(model.optimizer.learning_rate)", PyInputMode.SCRIPT);

    }


    @Test
    public void setAndGetGlobalPythonVariables() throws Exception {

        System.setProperty("jpy.debug", "true");
        System.setProperty("jpy.jdlLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jdl.so");
        System.setProperty("jpy.pythonLib", "/home/fmilo/anaconda2/envs/h2o/lib/libpython2.7.so");
        System.setProperty("jpy.jpyLib", "/home/fmilo/anaconda2/envs/h2o/lib/python2.7/site-packages/jpy.so");

        PyLib.startPython();
        PyLib.execScript("paramInt = 123");
        PyLib.execScript("paramStr = 'abc'");
        PyModule mainModule = PyModule.getMain();
        PyObject paramIntObj = mainModule.getAttribute("paramInt");
        PyObject paramStrObj = mainModule.getAttribute("paramStr");
        int paramIntValue = paramIntObj.getIntValue();
        String paramStrValue = paramStrObj.getStringValue();


        InputStream in = new FileInputStream(expandPath("~/workspace/h2o-3/h2o-algos/src/test/java/hex/deepwater/k_means.py"));
        String code = IOUtils.toString(in, StandardCharsets.UTF_8);
        PyLib.execScript(code);
        PyObject result = mainModule.callMethod("multiply", 2, 3 );



        // values from http://mnemstudio.org/clustering-k-means-example-1.htm
        PyObject kMeans = mainModule.callMethod("TFKMeansCluster", new double[][]{
                        new double[]{1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5},
                        new double[]{1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5},
                }, 2 );

        PyObject centroids = kMeans.call("__getitem__", 0);
        PyObject assignments = kMeans.call("__getitem__", 1);

        /////////////////////////////////////////////////
        assertEquals(6, result.getIntValue());
        assertEquals(123, paramIntValue);
        assertEquals("abc", paramStrValue);
        Double[] centroids_java = centroids.call("__getitem__", 0).getObjectArrayValue(Double.class);
        Assert.assertArrayEquals(new Double[]{1. ,  1.5,  3. ,  5. ,  3.5,  4.5,  3.5}, centroids_java);
        assertEquals(0, assignments.call("__getitem__",0).getIntValue());// (Integer.class));
        //Assert.assertArrayEquals(new Integer[]{0,1}, assignments.getIntValue());// (Integer.class));

        /////////////////////////////////////////////////

        //PyLib.stopPython();
    }*/


    private static class ImageParams {
        private final String imagePath;
        private final int input_height;
        private final int input_width;
        private final float input_mean;
        private final float input_std;

        private ImageParams(String imagePath, int input_height, int input_width, float input_mean, float input_std) {
            this.imagePath = imagePath;
            this.input_height = input_height;
            this.input_width = input_width;
            this.input_mean = input_mean;
            this.input_std = input_std;
        }

        public String getImagePath() {
            return imagePath;
        }

        public int getInput_height() {
            return input_height;
        }

        public int getInput_width() {
            return input_width;
        }

        public float getInput_mean() {
            return input_mean;
        }

        public float getInput_std() {
            return input_std;
        }
    }
}
