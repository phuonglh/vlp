package dca;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

/* 
* CompliantNode refers to a node that follows the rules (not malicious)
* <p>
* phuonglh@gmail.com
* 
*/
public class CompliantNode implements Node {
    boolean[] followees;
    Set<Transaction> pendingTransactions;
    // tx -> number of compliant senders that propose the tx
    Map<Transaction, Integer> counter = new HashMap<>(8);

    public CompliantNode(double p_graph, double p_malicious, double p_txDistribution, int numRounds) {
        // IMPLEMENT THIS
    }

    public void setFollowees(boolean[] followees) {
        // IMPLEMENT THIS
        this.followees = followees;
    }

    public void setPendingTransaction(Set<Transaction> pendingTransactions) {
        // IMPLEMENT THIS
        this.pendingTransactions = pendingTransactions;
    }

    public Set<Transaction> sendToFollowers() {
        // IMPLEMENT THIS
        if (counter.isEmpty())
            return pendingTransactions;
        List<Transaction> selection = new ArrayList<>();
        List<Entry<Transaction, Integer>> list = new ArrayList<>(counter.entrySet());
        list.sort(Entry.comparingByValue());
        Collections.reverse(list);

        for (Entry<Transaction, Integer> entry : list) {
            if (entry.getValue() >= 1 && entry.getKey().id != 0)
                selection.add(entry.getKey());
        }

        int n = Math.min(selection.size(), 200); // this is the key value: 200 ==> 82/100 points.
        return new HashSet<>(selection.subList(0, n));
    }

    public void receiveFromFollowees(Set<Candidate> candidates) {        
        // IMPLEMENT THIS
        // update this node internal transaction counter
        counter.clear();
        for (Candidate candidate: candidates) {
            Transaction tx = candidate.tx;
            if (followees[candidate.sender]) {
                if (!counter.containsKey(tx)) 
                    counter.put(tx, 0);
                counter.put(tx, counter.get(tx) + 1);
            }
        }
    }
}
