package vlp.idx;

import java.io.Serializable;
import java.util.Date;

/**
 * An interesting quote.
 */
public class Quote implements Serializable {
  private static final long serialVersionUID = -6750085006357533626L;
  private String topic;
  private String author;
  private String content;
  private Date date;

  public Quote(String topic, String author, String content) {
    this.topic = topic;
    this.author = author;
    this.content = content;
    this.date = new Date();
  }

  public Date getDate() {
    return date;
  }

  public String getAuthor() {
    return author;
  }

  public String getContent() {
    return content;
  }

  public String getTopic() {
    return topic;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(128);
    sb.append('[');
    sb.append(topic);
    sb.append(", ");
    sb.append(content);
    sb.append(", ");
    sb.append(author);
    sb.append(']');
    return sb.toString();
  }
}
