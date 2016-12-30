package deepwater.backends;
import deepwater.datasets.ImageDataSet;

import java.net.URI;

public interface BackendAPIV1 {

    void delete(BackendModel m);

    BackendModel buildNet(ImageDataSet dataset, RuntimeOptions opts,
                          BackendParams backend_params, int num_classes, String name);

    BackendSession createSession(RuntimeOptions options) throws BackendException;

    void deleteSession(BackendSession session) throws BackendException;

    BackendModel createModel(Tensor[] options) throws BackendException;

    void saveModel(BackendModel m, String model_path) throws BackendException;

    void loadModelVariables(BackendModel m, URI uri) throws BackendException;

    void saveModelVariables(BackendModel m, URI uri) throws BackendException;

    Tensor[] getModelParameters(BackendModel m) throws BackendException;

    void setModelParameters(BackendModel m, Tensor[] parameters) throws BackendException;

    Tensor[] fit(BackendModel m, Tensor[] data, Tensor[] label, Tensor[] options) throws BackendException;

    Tensor[] score(BackendModel m, Tensor[] data, Tensor[] label, Tensor[] options) throws BackendException;

}
