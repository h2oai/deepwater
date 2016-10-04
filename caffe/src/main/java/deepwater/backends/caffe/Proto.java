package deepwater.backends.caffe;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.caffe.*;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

interface Proto extends Externalizable {
  default void writeExternal(ObjectOutput out) {
    // TODO switch to coded streams for large messages
    if (this instanceof Message) {
      Message message = (Message) this;
      BytePointer bytes = message.SerializeAsString();
      try {
        out.writeInt((int) bytes.limit());
        out.write(bytes.getStringBytes());
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
      bytes.close();
    } else
      throw new RuntimeException("Cannot write " + getClass());
  }

  default void readExternal(ObjectInput in) {
    if (this instanceof Message) {
      Message message = (Message) this;
      try {
        byte[] array = new byte[in.readInt()];
        in.readFully(array);
        message.ParseFromString(new BytePointer(array));
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    } else
      throw new RuntimeException("Cannot read " + getClass());
  }
}
