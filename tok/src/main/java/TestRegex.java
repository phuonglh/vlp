import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TestRegex {
  public static void main(String[] args) throws Exception {
    List<String> lines = Files.readAllLines(Paths.get("/home/phuonglh/vlp/tok/src/main/resources/regexp.txt"));
    String url = lines.get(0);

    Pattern p = Pattern.compile(url);
    Matcher matcher = p.matcher("https://vnexpress.net/lam-ham-chui-tai-nut-giao-vanh-dai-3-5-voi-dai-lo-thang-long-4511997.html");
    System.out.println(matcher.matches());
    
    Matcher m2 = p.matcher("http://tuoitre.vn/");
    System.out.println(m2.matches());

    Matcher m3 = p.matcher("http//tuoitre.vn/");
    System.out.println(m3.matches());

  }
}
