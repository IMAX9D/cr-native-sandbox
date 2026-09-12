package royale.nativehost;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import org.json.JSONArray;
import org.json.JSONObject;

/** Experimental MessagePack response framing; no game-state fields are removed. */
public final class BinaryWire {
    private static final int MAX=64*1024*1024;
    private static final class LimitedBytes extends ByteArrayOutputStream {
        @Override public synchronized void write(int b) {
            if(count>=MAX)throw new IllegalArgumentException("binary frame limit");super.write(b);
        }
        @Override public synchronized void write(byte[] b,int off,int len) {
            if(len<0 || len>MAX-count)throw new IllegalArgumentException("binary frame limit");super.write(b,off,len);
        }
    }
    public static byte[] encode(String json) throws Exception {
        LimitedBytes bytes=new LimitedBytes();DataOutputStream out=new DataOutputStream(bytes);
        value(out,new JSONObject(json),0);out.flush();return bytes.toByteArray();
    }
    private static void value(DataOutputStream out,Object v,int depth) throws Exception {
        if(depth>64)throw new IllegalArgumentException("binary nesting limit");
        if(v==null || v==JSONObject.NULL){out.writeByte(0xc0);return;}
        if(v instanceof Boolean){out.writeByte(((Boolean)v)?0xc3:0xc2);return;}
        if(v instanceof Byte || v instanceof Short || v instanceof Integer || v instanceof Long) {
            long n=((Number)v).longValue();
            if(n>=-32 && n<=127)out.writeByte((int)n);
            else if(n>=-128 && n<=127){out.writeByte(0xd0);out.writeByte((int)n);}
            else if(n>=-32768 && n<=32767){out.writeByte(0xd1);out.writeShort((int)n);}
            else if(n>=Integer.MIN_VALUE && n<=Integer.MAX_VALUE){out.writeByte(0xd2);out.writeInt((int)n);}
            else {out.writeByte(0xd3);out.writeLong(n);}return;
        }
        if(v instanceof Number) {
            double d=((Number)v).doubleValue();if(Double.isNaN(d)||Double.isInfinite(d))throw new IllegalArgumentException("nonfinite binary value");
            out.writeByte(0xcb);out.writeDouble(d);return;
        }
        if(v instanceof String) {
            byte[] s=((String)v).getBytes(StandardCharsets.UTF_8);
            if(s.length<32)out.writeByte(0xa0|s.length);
            else if(s.length<256){out.writeByte(0xd9);out.writeByte(s.length);}
            else if(s.length<65536){out.writeByte(0xda);out.writeShort(s.length);}
            else{out.writeByte(0xdb);out.writeInt(s.length);}out.write(s);return;
        }
        if(v instanceof JSONArray) {
            JSONArray a=(JSONArray)v;
            if(a.length()<16)out.writeByte(0x90|a.length());
            else if(a.length()<65536){out.writeByte(0xdc);out.writeShort(a.length());}
            else{out.writeByte(0xdd);out.writeInt(a.length());}
            for(int i=0;i<a.length();i++)value(out,a.get(i),depth+1);return;
        }
        if(v instanceof JSONObject) {
            JSONObject o=(JSONObject)v;
            if(o.length()<16)out.writeByte(0x80|o.length());
            else if(o.length()<65536){out.writeByte(0xde);out.writeShort(o.length());}
            else{out.writeByte(0xdf);out.writeInt(o.length());}
            Iterator<String> keys=o.keys();while(keys.hasNext()){String k=keys.next();value(out,k,depth+1);value(out,o.get(k),depth+1);}return;
        }
        throw new IllegalArgumentException("unsupported binary value");
    }
}
