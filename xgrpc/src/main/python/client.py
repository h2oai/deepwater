from __future__ import print_function

import six
import grpc

import grpc_service_pb2 as pb

import numpy as np

from utils import convert_tensor, convert_feeds, convert_tensors


def check(status):
    if status.ok:
        return
    raise Exception(status.message)


def convert_value(value):
    pv = pb.ParamValue()
    if isinstance(value, six.integer_types):
        pv.i = value
    elif isinstance(value, six.string_types):
        pv.s = value
    elif isinstance(value, float):
        pv.f = value
    else:
        raise ValueError("not supported type:" + type(value))
    return pv


def encode_params(options):
    encoded = {}
    for key, value in six.iteritems(options):
        encoded[key] = convert_value(value)
    return encoded


def test_create_delete_session():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb.DeepWaterTrainBackendStub(channel)
    options = encode_params({
    })

    req = pb.CreateSessionRequest()
    response = stub.CreateSession(req)
    check(response.status)

    session = response.session

    response = stub.CreateModel(pb.CreateModelRequest(
        modelName="mlp",
        session=session,
        params=options,
    ))
    check(response.status)

    response = stub.DeleteSession(pb.DeleteSessionRequest(
        session=session,
    ))
    check(response.status)


def test_create_save_restore_delete_model():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb.DeepWaterTrainBackendStub(channel)
    options = encode_params({
    })

    req = pb.CreateSessionRequest()
    response = stub.CreateSession(req)
    check(response.status)

    session = response.session

    response = stub.CreateModel(pb.CreateModelRequest(
        modelName="mlp",
        session=session,
        params=options,
    ))
    check(response.status)

    response = stub.SaveModel(pb.SaveModelRequest(
        model=response.model,
        session=session,
        params=encode_params({
            'path': "/tmp/model.store.pb"
        })
    ))
    check(response.status)

    response = stub.LoadModel(pb.LoadModelRequest(
        session=session,
        params=encode_params({
            'path': "/tmp/model.store.pb"
        })
    ))
    check(response.status)
    model = response.model

    response = stub.SaveModelVariables(pb.SaveModelVariablesRequest(
        model=model,
        session=session,
        params=encode_params({
            'path': "/tmp/model.variables.store.pb"
        })
    ))
    check(response.status)

    response = stub.LoadModelVariables(pb.LoadModelVariablesRequest(
        model=model,
        session=session,
        params=encode_params({
            'path': "/tmp/model.variables.store.pb"
        })
    ))
    check(response.status)

    response = stub.DeleteSession(pb.DeleteSessionRequest(
        session=session,
    ))
    check(response.status)

    # restart the session and reload the previous saved model
    req = pb.CreateSessionRequest()
    response = stub.CreateSession(req)
    check(response.status)

    session = response.session
    response = stub.LoadModel(pb.LoadModelRequest(
        session=session,
        params=encode_params({
            'path': "/tmp/model.store.pb"
        })
    ))
    check(response.status)

    response = stub.LoadModelVariables(pb.LoadModelVariablesRequest(
        model=response.model,
        session=session,
        params=encode_params({
            'path': "/tmp/model.variables.store.pb"
        })
    ))
    check(response.status)

    # Delete the session
    response = stub.DeleteSession(pb.DeleteSessionRequest(
        session=session,
    ))
    check(response.status)


def test_train_models():
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    channel = grpc.insecure_channel('localhost:50051')
    stub = pb.DeepWaterTrainBackendStub(channel)

    options = encode_params({
    })

    req = pb.CreateSessionRequest()
    response = stub.CreateSession(req)
    check(response.status)

    session = response.session

    response = stub.CreateModel(pb.CreateModelRequest(
        modelName="lenet",
        session=session,
        params=options,
    ))
    check(response.status)

    model = response.model

    for i in six.moves.range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(128)

        feeds = [
            convert_tensor(batch_xs, "batch_image_input"),
            convert_tensor(batch_ys, "categorical_labels")
        ]

        fetches = [
            convert_tensor(np.array([]), 'categorical_logits'),
            convert_tensor(np.array([]), 'train'),
            convert_tensor(np.array([]), 'total_loss'),
            convert_tensor(np.array([]), 'global_step'),
            convert_tensor(np.array([]), 'accuracy')
        ]

        response = stub.Execute(pb.ExecuteRequest(
            session=session,
            model=model,
            fetches=fetches,
            feeds=feeds,
        ))
        check(response.status)

        values = convert_feeds(response.fetches)
        print(values['accuracy'].mean())
        print(values['total_loss'].mean())
        print(values['categorical_logits'].sum())
        print(values['global_step'])


if __name__ == '__main__':
    test_train_models()
    test_create_delete_session()
    test_create_save_restore_delete_model()
