package vlp.con;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.function.Consumer;

import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Treebank;

public class TreebankReader {

  static class TreeConsumer implements Consumer<Tree> {
    PrintWriter pw;

    public TreeConsumer(PrintWriter pw) {
      this.pw = pw;
    }

    @Override
    public void accept(Tree t) {
      pw.write(t.toString());
      pw.append("\n");
    }    
  }

  public static void main(String[] args) throws IOException {
    Treebank tb = new DiskTreebank();
    tb.loadPath("/Users/phuonglh/vlp/dat/con/file_3.txt");

    PrintWriter pw = new PrintWriter(new FileWriter("dat/3.txt"));

    tb.forEach(new TreeConsumer(pw));
    pw.close();
    System.out.println("Number of trees = " + tb.size());
  }
}
