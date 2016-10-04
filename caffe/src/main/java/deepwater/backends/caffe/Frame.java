package deepwater.backends.caffe;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.caffe.Datum;

public interface Frame {
  String[] names();
  Class[] types();
  Cursor cursor();

  interface Cursor {
    Object[] cells();
    void next();
  }

  class LMDB implements Frame {
    caffe.DB _db;

    public LMDB(String path) {
      _db = caffe.GetDB(caffe.DataParameter_DB_LMDB);
      _db.Open(path, caffe.READ);
    }

    @Override
    public String[] names() {
      return new String[]{"data", "label"};
    }

    @Override
    public Class[] types() {
      return new Class[]{byte[].class, int.class};
    }

    @Override
    public Cursor cursor() {
      final caffe.Cursor cursor = _db.NewCursor();
      final Datum datum = new Datum();
      cursor.SeekToFirst();

      return new Cursor() {
        @Override
        public Object[] cells() {
          datum.ParseFromString(cursor.value());
          BytePointer data = datum.data();
          byte[] bytes = new byte[(int) data.limit()];
          data.get(bytes);
          Object[] cells = new Object[2];
          cells[0] = bytes;
          cells[1] = datum.label();
          return cells;
        }

        @Override
        public void next() {
          cursor.Next();
          if (!cursor.valid())
            cursor.SeekToFirst();
        }
      };
    }
  }
}
