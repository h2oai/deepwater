package org.bytedeco.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = caffe.class,
    target = "org.bytedeco.javacpp.americano",
    value = {
        @Platform(
            includepath = {"include"},
            include = {"americano.hpp"}
        )
    }
)
public class americano implements InfoMapper {
  public void map(InfoMap infoMap) {
    infoMap.put(new Info("std::vector<caffe::FloatNCCL*>").pointerTypes("NCCLVector").define());
  }
}
