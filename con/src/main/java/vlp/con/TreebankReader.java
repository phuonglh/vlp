// package vlp.con;

// import java.io.FileWriter;
// import java.io.IOException;
// import java.io.PrintWriter;
// import java.io.StringReader;
// import java.util.function.Consumer;
// import java.util.function.Function;

// import edu.stanford.nlp.trees.DiskTreebank;
// import edu.stanford.nlp.trees.PennTreeReader;
// import edu.stanford.nlp.trees.Tree;
// import edu.stanford.nlp.trees.TreeLeafLabelTransformer;
// import edu.stanford.nlp.trees.TreeReader;
// import edu.stanford.nlp.trees.TreeTransformer;
// import edu.stanford.nlp.trees.Treebank;

// public class TreebankReader {

//   static class LeafReplacement implements Function<String, String> {
//     @Override
//     public String apply(String t) {
//       System.out.println(t);
//       return t.replaceAll("\\s+", "_").toLowerCase();
//     }
//   }

//   static class TreeConsumer implements Consumer<Tree> {
//     PrintWriter pw;
//     TreeTransformer tt = new TreeLeafLabelTransformer(new LeafReplacement());

//     public TreeConsumer(PrintWriter pw) {
//       this.pw = pw;
//     }
//     @Override
//     public void accept(Tree t) {
//       pw.write(t.transform(tt).toString());
//       pw.append("\n");
//     }    
//   }

//   public static void main(String[] args) throws IOException {
//     Treebank tb = new DiskTreebank();
//     tb.loadPath("../dat/con/file_3.2.txt");

//     PrintWriter pw = new PrintWriter(new FileWriter("dat/3.2.txt"));

//     tb.forEach(new TreeConsumer(pw));
//     pw.close();
//     System.out.println("Number of trees = " + tb.size());
//     // StringReader st = new StringReader("(S (NP (N He he)) (VP (V goes)))");
//     // TreeReader tr = new PennTreeReader(st);
//     // Tree tree = tr.readTree();
//     // tree.pennPrint();
//     // tr.close();
//   }
// }
