package deepwater.backends.tensorflow.models;

import com.google.gson.Gson;
import com.google.protobuf.ProtocolStringList;
import deepwater.backends.tensorflow.TensorflowMetaModel;
import org.bytedeco.javacpp.tensorflow;
import org.tensorflow.framework.*;
import org.tensorflow.framework.OpDef;
import org.tensorflow.framework.OpList;
import org.tensorflow.util.SaverDef;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static org.bytedeco.javacpp.tensorflow.*;
import static org.tensorflow.framework.CollectionDef.*;


public class ModelFactory {

    static public void debugJar() throws IOException {
        CodeSource src = ModelFactory.class.getProtectionDomain().getCodeSource();
        List<String> list = new ArrayList<>();

        if (src != null) {
            URL jar = src.getLocation();
            ZipInputStream zip = new ZipInputStream(jar.openStream());
            ZipEntry ze = null;

            while ((ze = zip.getNextEntry()) != null) {
                String entryName = ze.getName();
                list.add(entryName);
                System.out.println(list);
            }

        }

    }

    static public TensorflowModel LoadModel(String modelName) throws FileNotFoundException {
        String resourceModelName = convertToCanonicalName(modelName);
        String resourceMetaModelName = convertToCanonicalMetaName(modelName);
        tensorflow.GraphDef graph_def = extractGraphDefinition(resourceModelName);
        TensorflowMetaModel meta = extractMetaModel(resourceMetaModelName);
        return new TensorflowModel(meta, graph_def);
    }

    private static TensorflowMetaModel extractMetaModel(String resourceMetaModelName) throws FileNotFoundException {
        Gson g = new Gson();
        InputStream in = ModelFactory.class.getResourceAsStream("/"+resourceMetaModelName);
        if (in == null){
            String path = "/home/fmilo/workspace/deepwater/tensorflow/src/main/resources/" + resourceMetaModelName;
            in =  new FileInputStream(path);
        }
        Reader reader = null;
        try {
            reader = new InputStreamReader(in, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        return g.fromJson(reader, TensorflowMetaModel.class);
    }

    private static tensorflow.GraphDef extractGraphDefinition(String resourceModelName) {
        tensorflow.GraphDef graph_def = new tensorflow.GraphDef();
        try {
            //debugJar();
            String path;
            InputStream in = ModelFactory.class.getResourceAsStream("/"+resourceModelName);
            if (in != null) {
                path = saveToTempFile(in);
            } else {
                // FIXME: for some reason inside idea it does not work
                path = "/home/fmilo/workspace/deepwater/tensorflow/src/main/resources/" + resourceModelName;
            }
            checkStatus(ReadBinaryProto(Env.Default(), path, graph_def));
        } catch (IOException e) {
            e.printStackTrace();
            //throw new InvalidArgumentException(new String[]{"could not load model " + model_name});
        }
        return graph_def;
    }


    public static TensorflowModel readMetaGraph(String metaPath) throws Exception {
        MetaGraphDef metaGraphDef = MetaGraphDef.parseFrom(new FileInputStream(metaPath));
        org.tensorflow.framework.GraphDef graphDef = metaGraphDef.getGraphDef();
        // Tags
        for( String tag: metaGraphDef.getMetaInfoDef().getTagsList()) {
            System.out.println(tag);
        }


        OpList ops = metaGraphDef.getMetaInfoDef().getStrippedOpList();
        for( OpDef op: ops.getOpList()){
            System.out.println("stripped op: "+ op.getName());
        }

        TensorflowMetaModel meta = new TensorflowMetaModel();

        // SaverDef
        SaverDef saver = metaGraphDef.getSaverDef();
        meta.save_filename = saver.getFilenameTensorName();
        meta.save_op = saver.getSaveTensorName();
        meta.restore_op = saver.getRestoreOpName();


        meta.summary_op = getFirstOperationFromCollection(metaGraphDef, "summaries");
        meta.predict_op = getFirstOperationFromCollection(metaGraphDef, "logits");
        meta.train_op =  getFirstOperationFromCollection(metaGraphDef, "train");
        //getOperationsFromCollection(metaGraphDef, "variables");
        //getOperationsFromCollection(metaGraphDef, "trainable_variables");

        for(Map.Entry<String, SignatureDef> entry: metaGraphDef.getSignatureDefMap().entrySet()) {

            System.out.println("signature: " + entry.getKey());
            for(Map.Entry<String, TensorInfo> input: entry.getValue().getInputsMap().entrySet() ) {
                System.out.println("\tinput: " + input.getKey());
            }

            for(Map.Entry<String, TensorInfo> output: entry.getValue().getOutputsMap().entrySet() ) {
                System.out.println("\toutput: " + output.getKey());
            }

        }

        for(String name: getAllCollections(metaGraphDef)){
            System.out.println("collection: " + name);
        }

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        graphDef.writeTo(output);

        String path = saveToTempFile(new ByteArrayInputStream(output.toByteArray()));

        tensorflow.GraphDef gdef = new tensorflow.GraphDef();
        Status status = ReadBinaryProto(tensorflow.Env.Default(), path, gdef);
        checkStatus(status);
        return new TensorflowModel(meta, gdef);
    }

    private static String[] getAllCollections(MetaGraphDef metaGraphDef) throws Exception {
        Set<String> keys = metaGraphDef.getCollectionDefMap().keySet();
        String[] results = new String[keys.size()];
        keys.toArray(results);
        return results;

    }

    private static String getFirstOperationFromCollection(MetaGraphDef metaGraphDef, String collectionName) throws Exception {
        String[] ops = getOperationsFromCollection(metaGraphDef, collectionName);
        if (ops.length > 0){
            return ops[0];
        }
        return "";
    }

    private static String[] getOperationsFromCollection(MetaGraphDef metaGraphDef, String collectionName) throws Exception {
        for(Map.Entry<String, CollectionDef> entry: metaGraphDef.getCollectionDefMap().entrySet()) {
            if(entry.getValue().equals(collectionName)){
               KindCase kase = entry.getValue().getKindCase();

               switch(kase) {
                case NODE_LIST:
                    ProtocolStringList vlist = entry.getValue().getNodeList().getValueList();
                    String [] result = new String[vlist.size()];
                    vlist.toArray(result);
                    return result;
                default:
                    throw new Exception("invalid collection format.");
                case KIND_NOT_SET:
                    break;
               }
            }
        }
        return new String[]{};
    }

    private static String saveToTempFile(InputStream in) throws IOException {
        String path;
        File temp = File.createTempFile("tempfile", ".tmp");
        BufferedWriter bw = new BufferedWriter(new FileWriter(temp));
        bw.close();
        Files.copy(in, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
        path = temp.getAbsolutePath();
        return path;
    }

    private static String convertToCanonicalMetaName(String model_name) {
        return model_name.toLowerCase() + ".meta";
    }

    private static String convertToCanonicalName(String model_name) {
        return model_name.toLowerCase() + ".pb";
    }

    static void checkStatus(Status status) {
        if (!status.ok()) {
            throw new InternalError(status.error_message().getString());
        }
    }
}
