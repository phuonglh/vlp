package vlp.idx;

import java.util.Date;

/**
 * phuonglh@gmail.com
 * <p/>
 * 2019-06-24, 18:47
 */
public class News {
  private String url;
  private String content;
  private Date date;

  public News(String url, String content, Date date) {
    this.url = url;
    this.content = content;
    this.date = date;
  }

  public News(String url, String content) {
    this(url, content, new Date());
  }

  public String getContent() {
    return content;
  }

  public Date getDate() {
    return date;
  }

  public String getUrl() {
    return url;
  }
}
