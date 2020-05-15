package vlp.idx;


import java.util.Date;

/**
 * A piece of text processed by the system.
 */

public class Text {
  private String task;
  private String content;
  private Date date;

  public Text(String task, String content) {
    this(task, content, new Date());
  }

  public Text(String task, String content, Date date) {
    this.task = task;
    this.content = content;
    this.date = date;
  }

  public String getContent() {
    return content;
  }

  public String getTask() {
    return task;
  }

  public Date getDate() {
    return date;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('[');
    sb.append(task);
    sb.append(", ");
    sb.append(content);
    sb.append(", ");
    sb.append(date);
    sb.append(']');
    return sb.toString();
  }
}
