package vlp.idx;

import org.apache.http.HttpHost;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.bulk.BulkRequest;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.common.unit.TimeValue;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.sort.SortOrder;

import javax.json.*;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder;


/**
 * Elastic Search Indexer.
 */
public class Indexer {
  static RestHighLevelClient client = new RestHighLevelClient(RestClient.builder(new HttpHost("vlp.group", 9200, "http")));
  static final Logger logger = Logger.getLogger(Indexer.class.getName());

  /**
   * Index one text sample.
   * @param text a text
   * @throws IOException
   */
  public static void indexOne(Text text) throws IOException {
    XContentBuilder builder = jsonBuilder().startObject();
    builder.field("task", text.getTask()).field("content", text.getContent()).field("date", text.getDate());
    builder.endObject();
    IndexRequest request = new IndexRequest("texts").source(builder).timeout(TimeValue.timeValueSeconds(1));
    client.indexAsync(request, RequestOptions.DEFAULT, new ActionListener<IndexResponse>() {
      @Override
      public void onResponse(IndexResponse indexResponse) {
        logger.info("Success indexing: " + text.toString());
      }
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure indexing: " + text.toString());
      }
    });
  }

  /**
   * Index one news sample.
   * @param news a news
   * @throws IOException
   */
  public static void indexOne(News news) throws IOException {
    XContentBuilder builder = jsonBuilder().startObject();
    builder.field("url", news.getUrl()).field("content", news.getContent()).field("date", news.getDate());
    builder.endObject();
    IndexRequest request = new IndexRequest("news").source(builder).timeout(TimeValue.timeValueSeconds(1));
    client.indexAsync(request, RequestOptions.DEFAULT, new ActionListener<IndexResponse>() {
      @Override
      public void onResponse(IndexResponse indexResponse) {
        logger.info("Success indexing: " + news.getUrl());
      }
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure indexing: " + news.getUrl());
      }
    });
  }

  /**
   * Index one quote sample.
   * @param quote a quote
   * @throws IOException
   */
  public static void indexOne(Quote quote) throws IOException {
    XContentBuilder builder = jsonBuilder().startObject();
    builder.field("topic", quote.getTopic()).field("content", quote.getContent())
        .field("author", quote.getAuthor()).field("date", quote.getDate());
    builder.endObject();
    IndexRequest request = new IndexRequest("quote").source(builder).timeout(TimeValue.timeValueSeconds(1));
    client.indexAsync(request, RequestOptions.DEFAULT, new ActionListener<IndexResponse>() {
      @Override
      public void onResponse(IndexResponse indexResponse) {
        logger.info("Success indexing: " + quote.getContent());
      }
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure indexing: " + quote.getContent());
      }
    });
  }

  /**
   * Index many text samples in one request.
   * @param texts a list of texts
   * @throws IOException
   */
  public static void indexManyTexts(List<Text> texts) throws IOException {
    BulkRequest request = new BulkRequest();
    for (Text text: texts) {
      XContentBuilder builder = jsonBuilder().startObject();
      builder.field("task", text.getTask()).field("content", text.getContent()).field("date", text.getDate());
      builder.endObject();
      request.add(new IndexRequest("texts").source(builder));
    }
    client.bulkAsync(request, RequestOptions.DEFAULT, new ActionListener<BulkResponse>() {
      @Override
      public void onResponse(BulkResponse bulkItemResponses) {}
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure in indexing " + texts.size() + " texts.");
      }
    });
  }

  /**
   * Index many news samples in one request.
   * @param news a list of texts
   * @throws IOException
   */
  public static void indexManyNews(List<News> news) throws IOException {
    BulkRequest request = new BulkRequest();
    for (News ns: news) {
      XContentBuilder builder = jsonBuilder().startObject();
      builder.field("url", ns.getUrl()).field("content", ns.getContent()).field("date", ns.getDate());
      builder.endObject();
      request.add(new IndexRequest("news").source(builder));
    }
    client.bulkAsync(request, RequestOptions.DEFAULT, new ActionListener<BulkResponse>() {
      @Override
      public void onResponse(BulkResponse bulkItemResponses) {
        logger.info("Success in indexing " + news.size() + " news.");
      }
      @Override
      public void onFailure(Exception e) {
        e.printStackTrace();
        logger.info("Failure in indexing " + news.size() + " news.");
      }
    });
  }

  /**
   * Index all news samples from a JSON input file (imported from vitk-vqc PJ). Note that
   * we need to filter existing URLs from the index.
   * @param jsonInputPath an input file in JSON format.
   * @throws IOException
   */
  public static void indexManyNews(String jsonInputPath) throws IOException {
    Set<String> ignored = new HashSet<>();
    ignored.add("XE 360º");
    ignored.add("TỪ KHÓA");
    ignored.add("Xem thêm");
    ignored.add("Bình luận");
    ignored.add(".");
    // load the json file and construct a list of News.
    List<String> json = Files.readAllLines(Paths.get(jsonInputPath));
    List<News> news = json.stream().map(line -> {
      JsonReader reader = Json.createReader(new StringReader(line));
      JsonObject object = reader.readObject();
      String url = object.getString("url");
      JsonArray array = object.getJsonArray("sentences");
      StringBuilder content = new StringBuilder(1024);
      for (int j = 0; j < array.size(); j++) {
        String s = array.getString(j);
        if (!ignored.contains(s)) {
          content.append(s);
          content.append("\n");
        }
      }
      return new News(url, content.toString().trim());
    }).collect(Collectors.toList());
    // load the news index and build a set of existing URLs
    Set<String> existingURLs = new HashSet<>();
    // create a request
    SearchRequest request = new SearchRequest("news");
    SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
    searchSourceBuilder.query(QueryBuilders.matchAllQuery());
    searchSourceBuilder.size(10000);
    searchSourceBuilder.sort("date", SortOrder.DESC);
    request.source(searchSourceBuilder);
    // execute the request and get a response
    SearchResponse response = client.search(request, RequestOptions.DEFAULT);
    // extract and return result
    SearchHit[] hits = response.getHits().getHits();
    for (SearchHit hit: hits) {
      String url = hit.getSourceAsMap().get("url").toString();
      existingURLs.add(url);
    }
    // filter novel news
    List<News> novelNews = news.stream().filter(element -> !existingURLs.contains(element.getUrl()))
        .filter(element -> element.getContent().trim().length() >= 200 & !element.getContent().contains("<div") &&
            !element.getContent().contains("<table") && !element.getContent().contains("</p>"))
        .collect(Collectors.toList());

    logger.info("#(novelNews) = " + novelNews.size());
    // divide novel news into small chunks of 2000 samples
    int n = novelNews.size() / 2000;
    for (int i = 0; i < n - 1; i++) {
      indexManyNews(novelNews.subList(2000*i, 2000*(i+1)));
      try {
        Thread.sleep(5000);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
      logger.info("indexed: " + 2000*(i+1));
    }
    indexManyNews(novelNews.subList(n*2000, novelNews.size()));
  }

  /**
   * Index many news samples in one request.
   * @param quotes a list of texts
   * @throws IOException
   */
  public static void indexManyQuotes(List<Quote> quotes) throws IOException {
    BulkRequest request = new BulkRequest();
    for (Quote quote: quotes) {
      XContentBuilder builder = jsonBuilder().startObject();
      builder.field("topic", quote.getTopic()).field("content", quote.getContent())
          .field("author", quote.getAuthor()).field("date", quote.getDate());
      builder.endObject();
      request.add(new IndexRequest("quotes").source(builder));
    }
    client.bulkAsync(request, RequestOptions.DEFAULT, new ActionListener<BulkResponse>() {
      @Override
      public void onResponse(BulkResponse bulkItemResponses) {}
      @Override
      public void onFailure(Exception e) {
        logger.info("Failure in indexing " + quotes.size() + " quotes.");
      }
    });
  }

  public static void indexManyQuotes(String textInputPath) {
    try {
      List<String> lines = Files.readAllLines(Paths.get(textInputPath), StandardCharsets.UTF_8);
      List<Quote> quotes = lines.stream().map(line -> {
        String[] parts = line.split(";");
        return new Quote("Life", parts[1].trim(), parts[0].trim());
      }).collect(Collectors.toList());
      quotes.forEach(System.out::println);
      Indexer.indexManyQuotes(quotes);
      Thread.sleep(2000);
      client.close();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public static void close() {
    try {
      if (client != null)
        client.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws IOException, InterruptedException {
    indexManyNews("dat/txt/news.json");
    Thread.sleep(5000);
    Indexer.close();
    System.out.println("Done.");
  }
}
