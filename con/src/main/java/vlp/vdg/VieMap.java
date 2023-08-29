package vlp.vdg;

import java.io.Serializable;
import java.util.concurrent.ConcurrentHashMap;

/**
 * phuonglh
 */
public class VieMap extends ConcurrentHashMap<Character, Character> implements Serializable {

  public VieMap() {
    // lowercase characters
    put('à', 'a'); put('á', 'a'); put('ả', 'a'); put('ã', 'a'); put('ạ', 'a');
    put('â', 'a'); put('ầ', 'a'); put('ấ', 'a'); put('ẩ', 'a'); put('ẫ', 'a'); put('ậ', 'a');
    put('ă', 'a'); put('ằ', 'a'); put('ắ', 'a'); put('ẳ', 'a'); put('ẵ', 'a'); put('ặ', 'a');
    put('è', 'e'); put('é', 'e'); put('ẻ', 'e'); put('ẽ', 'e'); put('ẹ', 'e');
    put('ê', 'e'); put('ề', 'e'); put('ế', 'e'); put('ể', 'e'); put('ễ', 'e'); put('ệ', 'e');
    put('ò', 'o'); put('ó', 'o'); put('ỏ', 'o'); put('õ', 'o'); put('ọ', 'o');
    put('ì', 'i'); put('í', 'i'); put('ỉ', 'i'); put('ĩ', 'i'); put('ị', 'i');
    put('ô', 'o'); put('ồ', 'o'); put('ố', 'o'); put('ổ', 'o'); put('ỗ', 'o'); put('ộ', 'o');
    put('ơ', 'o'); put('ờ', 'o'); put('ớ', 'o'); put('ở', 'o'); put('ỡ', 'o'); put('ợ', 'o');
    put('ù', 'u'); put('ú', 'u'); put('ủ', 'u'); put('ũ', 'u'); put('ụ', 'u');
    put('ư', 'u'); put('ừ', 'u'); put('ứ', 'u'); put('ử', 'u'); put('ữ', 'u'); put('ự', 'u');
    put('ỳ', 'y'); put('ý', 'y'); put('ỷ', 'y'); put('ỹ', 'y'); put('ỵ', 'y');
    put('đ', 'd');
    // uppercase characters
    for (Character c: keySet()) {
      Character v = get(c);
      put(Character.toUpperCase(c), Character.toUpperCase(v));
    }
  }

  public boolean contains(char c) {
    return containsKey(c) || containsValue(c);
  }

  public boolean contains(String c) {
    return this.contains(c.charAt(0));
  }
}
