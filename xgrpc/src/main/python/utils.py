import six
import grpc_service_pb2 as pb
import numpy as np


def model_must_be_valid(model):
    pass

def convert_tensors(fetches, numpy_tensor_list):
    result = []
    for name, np_array in zip(fetches, numpy_tensor_list):
        if np_array is None:
            continue
        proto_tensor = convert_tensor(np_array, name)
        result.append(proto_tensor)
    return result

def convert_tensor(np_tensor, name=""):
    dim = [ pb.Shape.Dim(size=d) for d in np_tensor.shape ]
    return pb.Tensor(
        name=name,
        type=pb.DT_FLOAT32,
        shape=pb.Shape(dim=dim),
        float_value=np_tensor.reshape((-1,)).tolist()
    )

def convert_feeds(params):
    feeds = {}
    for p in params:
        assert isinstance(p, pb.Tensor)

        if p.type == pb.DT_FLOAT32:
            a = np.array(p.float_value)
        elif p.type == pb.DT_FLOAT64:
            a = np.array(p.double_value)
        elif p.type == pb.DT_INVALID:
            raise Exception("tensor type was not specified")
        else:
            raise Exception("tensor type was not specified")
        shape = [ d.size for d in p.shape.dim]
        a = a.reshape(shape)
        feeds[p.name] = a 
    return feeds


def convert_fetches(params):
    return [p.name for p in params]


def convert_feeds(params):
    feeds = {}
    for p in params:
        assert isinstance(p, pb.Tensor)

        if p.type == pb.DT_FLOAT32:
            a = np.array(p.float_value)
        elif p.type == pb.DT_FLOAT64:
            a = np.array(p.double_value)
        elif p.type == pb.DT_INVALID:
            raise Exception("tensor type was not specified")
        else:
            raise Exception("tensor type was not specified")
        shape = [ d.size for d in p.shape.dim]
        a = a.reshape(shape)
        feeds[p.name] = a 
    return feeds


def convert_map(params):
    "Convert from a pb request param to a python map"

    result = {}
    for key, param in params.items():
        field = param.WhichOneof('value')
        value = getattr(param, field)
        result[key] = value
    return result



