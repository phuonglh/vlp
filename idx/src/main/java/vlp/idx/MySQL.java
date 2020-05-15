package vlp.idx;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.*;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * phuonglh@gmail.com
 * <p/>
 * 2019-06-30, 14:43
 *
 * This utility populates the 'vlp' MySQL database with the old file `news.json'.
 *
 * First, create a database `vlp` on the server, create an account and its grant of access (local and/or remote).
 * Then create a table `url` on that database. Log on to MySQL:
 *
 * mysql -u vlp -p
 * use vlp;
 * create table (id int auto_increment, url varchar(255), primary key (id));
 *
 * Then read the file `news.json` and insert all URLs into table `url`. All these URLs were indexed in
 * the ElasticSearch server.
 *
 */
public class MySQL {
  static String connectionURL = "jdbc:mysql://vlp.group/vlp?user=vlp&password=";
  /**
   * Inserts a list of URLs into the `url` table of a MySQL server.
   * @param urls
   */
  public static void insert(Set<String> urls) {
    PreparedStatement statement = null;
    ResultSet result = null;
    try {
      Class.forName("com.mysql.cj.jdbc.Driver").newInstance();
      Connection connection = DriverManager.getConnection(connectionURL);
      String query = "INSERT INTO url (url) VALUES (?)";
      statement = connection.prepareStatement(query);
      for (String url: urls) {
        if (url.length() < 256) {
          statement.setString(1, url);
          statement.addBatch();
        }
      }
      statement.executeBatch();
      System.out.println("Done.");
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    } finally {
      if (result != null) {
        try {
          result.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
      if (statement != null) {
        try {
          statement.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
    }
  }

  /**
   * Gets all URLs in the MySQL database `url`.
   * @return a set of URLs.
   */
  public static Set<String> getURLs() {
    Set<String> urls = new HashSet<>();
    Statement statement = null;
    ResultSet result = null;
    try {
      Class.forName("com.mysql.cj.jdbc.Driver").newInstance();
      Connection connection = DriverManager.getConnection(connectionURL);
      statement = connection.createStatement();
      result = statement.executeQuery("SELECT url from url");
      while (result.next()) {
        String url = result.getString("url");
        urls.add(url);
      }
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    } finally {
      if (result != null) {
        try {
          result.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
      if (statement != null) {
        try {
          statement.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
    }
    return urls;
  }

  public static void populate(String jsonInputPath) {
    List<String> urls = new LinkedList<>();
    try {
      List<String> lines = Files.readAllLines(Paths.get(jsonInputPath));
      for (String json: lines) {
        JsonReader reader = Json.createReader(new StringReader(json));
        JsonObject object = reader.readObject();
        urls.add(object.getString("url"));
      }
    } catch (IOException e) {
      e.printStackTrace();
    }

    System.out.println("#(urls) = " + urls.size());
    for (int i = 0; i < 10; i++)
      System.out.println(urls.get(i));

    PreparedStatement statement = null;
    ResultSet result = null;
    final int batchSize = 1000;
    int count = 0;
    try {
      Class.forName("com.mysql.cj.jdbc.Driver").newInstance();

      Connection connection = DriverManager.getConnection(connectionURL);
      String query = "INSERT INTO url (url) VALUES (?)";
      statement = connection.prepareStatement(query);
      for (String url: urls) {
        statement.setString(1, url);
        statement.addBatch();
        if (++count % batchSize == 0)
          statement.executeBatch();
      }
      statement.executeBatch();
      System.out.println("Done.");
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    } finally {
      if (result != null) {
        try {
          result.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
      if (statement != null) {
        try {
          statement.close();
        } catch (SQLException e) {
          e.printStackTrace();
        }
      }
    }
  }

  public static void main(String[] args) {
//    populate("dat/fin/news.json");
    Set<String> urls = getURLs();
    System.out.println("#(urls) = " + urls.size());
  }
}
