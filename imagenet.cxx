
#include "imagenet.hpp"

class BufferFile {
 public :
  BufferFile(const std::string & file_path);
  int getLength() {return length_;}
  char* getBuffer() {return buffer_;}
  ~BufferFile();
 private:
  std::string file_path_;
  int length_;
  char* buffer_;
};

BufferFile::BufferFile(const std::string & file_path){
  file_path_ = file_path;
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  ifs.seekg(0, std::ios::end);
  length_ = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  buffer_ = new char[sizeof(char) * length_];
  ifs.read(buffer_, length_);
  ifs.close();
}

BufferFile::~BufferFile() {
  delete[] buffer_;
  buffer_ = NULL;
}

std::vector<std::string> loadSynset(const std::string & filename) {
  std::ifstream fi(filename.c_str());

  std::vector<std::string> output;

  std::string synset, lemma;
  while ( fi >> synset ) {
    getline(fi, lemma);
    output.push_back(lemma);
  }

  fi.close();

  return output;
}

ImageNative::ImageNative() {
  width = 224;
  height = 224;
  channels = 3;
  image_size = width * height * channels;
  dev_type = 1;
  dev_id = 0;
}

void ImageNative::loadInception() {
  synset = loadSynset(model_path_ + "/Inception/synset.txt");
  BufferFile json_data(model_path_ + "/Inception_BN-symbol.json");
  BufferFile param_data(model_path_ + "/Inception_BN-0039.params");
  BufferFile nd_buf(model_path_ + "/Inception/mean_224.nd");

  mx_uint nd_index = 0;
  mx_uint nd_len;
  mx_uint nd_ndim = 0;
  const char* nd_key = 0;
  const mx_uint* nd_shape = 0;

  MXNDListCreate((const char*)nd_buf.getBuffer(),nd_buf.getLength(),&nd_hnd, &nd_len);

  MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);

  const mx_uint input_shape_indptr[2] = {0, 4};
  const mx_uint input_shape_data[4] = {
    1,
    static_cast<mx_uint>(channels),
    static_cast<mx_uint>(width),
    static_cast<mx_uint>(height)
  };

  mx_uint num_input_nodes = 1;  // 1 for feedforward
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  MXPredCreate((const char*)json_data.getBuffer(),
               (const char*)param_data.getBuffer(),
               static_cast<size_t>(param_data.getLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);

}

const char * ImageNative::predict(float * image_data) {
  for (int i = 0; i < image_size; i++) {
    image_data[i] = image_data[i] - nd_data[i];
  }

  MXPredSetInput(pred_hnd, "data", image_data, image_size);

  MXPredForward(pred_hnd);

  mx_uint output_index = 0;
  mx_uint *shape = 0;
  mx_uint shape_len;

  MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

  std::vector<float> data(size);

  MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);

  std::string res;

  return res.c_str();
}

ImageNative::~ImageNative() {
  MXNDListFree(nd_hnd);
  MXPredFree(pred_hnd);
  delete[] nd_data;
}
