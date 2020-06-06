package vlp.vdg;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * phuonglh, 10/19/18, 00:43
 * <p>
 *   Split a long line into smaller ones, each less than a given 
 *   <code>maxLength</code>. This max length threshold can be measured 
 *   in characters or tokens.
 * </p>
 */
public class LineSlicer implements Serializable  {
  private int maxLength;
  
  public LineSlicer() {
    this(80);
  }
  
  public LineSlicer(int maxLength) {
    this.maxLength = maxLength;
  }
  
  public List<String> split(String data) {
    List<String> xs = new LinkedList<>();
    String s = data.trim();
    while (!s.isEmpty()) {
      int t = Math.min(s.length(), maxLength);
      if (t == maxLength) {
        while (t > 0 && s.charAt(--t) != ' ') ;
      }
      String u = s.substring(0, t);
      xs.add(u.trim());
      s = s.substring(t).trim();
    }
    return xs;
  }
  
  public List<String[]> split(String[] data) {
    List<String[]> xs = new LinkedList<>();
    int j = 0;
    while (j < data.length) {
      int t = Math.min(data.length, maxLength);
      xs.add(Arrays.copyOfRange(data, j, Math.min(data.length, j + t)));
      j += t;
    }
    return xs;
  }

  public static void main(String[] args) {
    LineSlicer slicer = new LineSlicer(16);
    String text = "Ý kiến đa chiều về nhà hát giao hưởng ở Thủ Thiêm";
    slicer.split(text).forEach(System.out::println);
    
    slicer = new LineSlicer(2);
    String[] data = new String[]{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"};
    List<String[]> parts = slicer.split(data);
    for (String[] p: parts)
      System.out.println(Arrays.asList(p));
  }
}
