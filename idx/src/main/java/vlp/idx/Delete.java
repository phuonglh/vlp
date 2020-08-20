package vlp.idx;


import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.Request;
import org.apache.http.HttpHost;
import org.elasticsearch.action.admin.indices.get.GetIndexRequest;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.util.stream.Collectors;


/**
 * Deletes all indices in the server. Use this utility with caution.
 * 
 * phuonglh@gmail.com
 * 
 */
public class Delete {
  static final String URL = "http://group.vlp:9200/";
  static final Set<String> kept = new HashSet<String>();

  public static void main(String[] args) throws Exception {
    kept.add("news");
    kept.add("texts");

    RestHighLevelClient client = new RestHighLevelClient(RestClient.builder(new HttpHost("vlp.group", 9200, "http")));
    Request request = new Request("GET", Delete.URL + "_cat/indices");
    InputStream inputStream = client.getLowLevelClient().performRequest(request).getEntity().getContent();
    
    List<String> indexes = new BufferedReader(new InputStreamReader(inputStream)).lines().collect(Collectors.toList());
    for (String line: indexes) {
      String[] parts = line.split("\\s+");
      String indexName = parts[2];
      if (!kept.contains(indexName)) {
        Request deleteReq = new Request("DELETE", Delete.URL + indexName);
        client.getLowLevelClient().performRequest(deleteReq);
        System.out.println(indexName + " deleted.");
      }
    }
    client.close();
  }
}