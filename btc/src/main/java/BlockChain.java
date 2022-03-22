import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

// Block Chain should maintain only limited block nodes to satisfy the functions
// You should not have all the blocks added to the block chain in memory 
// as it would cause a memory overflow.

public class BlockChain {
    public static final int CUT_OFF_AGE = 2;
    class Node {
        Block block;
        Node parent;
        List<Node> children;
        Node(Block block, Node parent) {
            this.block = block;
            this.parent = parent;
            this.children = new LinkedList<>();
        }
        Node(Block block) {
            this(block, null);
        }
    }
    Node chain;
    Map<Block, UTXOPool> uPoolMap = new HashMap<>();
    TransactionPool txPool = new TransactionPool();

    /**
     * create an empty block chain with just a genesis block. Assume {@code genesisBlock} is a valid
     * block
     */
    public BlockChain(Block genesisBlock) {
        // IMPLEMENT THIS
        chain = new Node(genesisBlock);
        uPoolMap.put(genesisBlock, new UTXOPool());
    }

    class Pair {
        Block block;
        int height;
        Pair(Block block, int height) {
            this.block = block;
            this.height = height;
        }
    }

    private Pair getMaxPair(Node node) {
        if (node.children.isEmpty())
            return new Pair(node.block, 1);
        else {
            Pair[] pairs = new Pair[node.children.size()];
            int maxHeightIndex = 0;
            for (int j = 0; j < pairs.length; j++) {
                pairs[j] = getMaxPair(node.children.get(j));
                if (pairs[maxHeightIndex].height < pairs[j].height) {
                    maxHeightIndex = j;
                }
            }
            return new Pair(pairs[maxHeightIndex].block, pairs[maxHeightIndex].height + 1);
        }
    }

    /** Get the maximum height block */
    public Block getMaxHeightBlock() {
        // IMPLEMENT THIS
        return getMaxPair(chain).block;
    }

    /** Get the UTXOPool for mining a new block on top of max height block */
    public UTXOPool getMaxHeightUTXOPool() {
        // IMPLEMENT THIS
        Block block = getMaxHeightBlock();
        if (!uPoolMap.containsKey(block))
            uPoolMap.put(block, new UTXOPool());
        return uPoolMap.get(block);
    }

    /** Get the transaction pool to mine a new block */
    public TransactionPool getTransactionPool() {
        // IMPLEMENT THIS
        return txPool;
    }

    /**
     * Add {@code block} to the block chain if it is valid. For validity, all transactions should be
     * valid and block should be at {@code height > (maxHeight - CUT_OFF_AGE)}.
     * 
     * <p>
     * For example, you can try creating a new block over the genesis block (block height 2) if the
     * block chain height is {@code <=
     * CUT_OFF_AGE + 1}. As soon as {@code height > CUT_OFF_AGE + 1}, you cannot create a new block
     * at height 2.
     * 
     * @return true if block is successfully added
     */
    public boolean addBlock(Block block) {
        // IMPLEMENT THIS
        if (block.getPrevBlockHash() == null)
            return false;
        // check validity
        TxHandler txHandler = new TxHandler(getMaxHeightUTXOPool());
        List<Transaction> txs = block.getTransactions();
        boolean isValidBlock = true;
        for (Transaction tx: txs)
            if (!txHandler.isValidTx(tx)) {
                isValidBlock = false;
                break;
            }
        if (!isValidBlock)
            return false;
        // removed added txs from the transaction pool
        for (Transaction tx: txs)
            txPool.removeTransaction(tx.getHash());

        // find nodes at height > maxHeight - CUT_OFF_AGE
        int maxHeight = getMaxPair(chain).height;
        System.out.println(maxHeight);
        Node p = chain;
        int height = 0;
        List<Node> candidates = new ArrayList<>();
        while (p != null && height <= maxHeight - CUT_OFF_AGE) {
            candidates.add(p);
            height++;
            if (p.children.isEmpty()) 
                p = null;
            else {
                int j = new Random().nextInt(p.children.size());
                p = p.children.get(j);
            }
        }
        // append the given block into one branch, here I deliberatly choose the last candidate
        if (candidates.isEmpty())
            candidates.add(chain);
        Node last = candidates.get(candidates.size()-1);
        last.children.add(new Node(block, last));
        // debug: print blockchain
        // 
        // update the genesis block
        chain = candidates.get(0);
        return true;
    }

    /** Add a transaction to the transaction pool */
    public void addTransaction(Transaction tx) {
        // IMPLEMENT THIS
        txPool.addTransaction(tx);
    }
}