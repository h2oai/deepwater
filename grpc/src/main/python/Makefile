

install:
	pip install -r requirements.txt

all:
	python -m grpc.tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/grpc_service.proto
