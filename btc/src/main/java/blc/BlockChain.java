package blc;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;

// Block Chain should maintain only limited block nodes to satisfy the functions
// You should not have all the blocks added to the block chain in memory 
// as it would cause a memory overflow.

public class BlockChain {
    public static final int CUT_OFF_AGE = 2;

    class Node {
        Block block;
        Node parent;
        List<Node> children;
        int height; 
        Node(Block block, Node parent) {
            this.block = block;
            this.parent = parent;
            this.children = new LinkedList<>();
            this.height = 0;
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
        Transaction tx = new Transaction(genesisBlock.getCoinbase());
        tx.addInput(genesisBlock.getCoinbase().getHash(), 0);
        genesisBlock.addTransaction(tx);
        uPoolMap.put(genesisBlock, new UTXOPool());
        updateUTXPool(genesisBlock);
    }

    private void updateUTXPool(Block block) {
        if (!uPoolMap.containsKey(block))
            uPoolMap.put(block, new UTXOPool());
        UTXOPool pool = uPoolMap.get(block);
        for (Transaction tx: block.getTransactions()) {
            List<Transaction.Output> outputs = tx.getOutputs();
            for (int j = 0; j < outputs.size(); j++) {
                UTXO utxo = new UTXO(tx.getHash(), j);
                pool.addUTXO(utxo, outputs.get(j));
            }
        }
    }

    private Node getMaxHeightNode() {
        Node maxHeightNode = chain;
        Queue<Node> queue = new LinkedBlockingQueue<>();
        queue.add(chain);
        while (!queue.isEmpty()) {
            Node p = queue.remove();
            if (p.height > maxHeightNode.height) {
                maxHeightNode = p;
            }
            for (Node node: p.children)
                queue.add(node);
        }
        return maxHeightNode;

    }

    /** Get the maximum height block */
    public Block getMaxHeightBlock() {
        // IMPLEMENT THIS
        return getMaxHeightNode().block;
    }

    /** Get the UTXOPool for mining a new block on top of max height block */
    public UTXOPool getMaxHeightUTXOPool() {
        // IMPLEMENT THIS
        return uPoolMap.get(getMaxHeightBlock());
    }

    /** Get the transaction pool to mine a new block */
    public TransactionPool getTransactionPool() {
        // IMPLEMENT THIS
        return txPool;
    }

    private boolean checkBlockHash(byte[] hash) {
        Set<Block> set = uPoolMap.keySet();
        ByteArrayWrapper w = new ByteArrayWrapper(hash);
        for (Block block: set) {
            if (new ByteArrayWrapper(block.getHash()).equals(w))
                return true;
        }
        return false;
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
        byte[] prevHash = block.getPrevBlockHash();
        if (prevHash == null) 
            return false;
        if (!checkBlockHash(prevHash))
            return false;
        int maxHeight = getMaxHeightNode().height;
        // traverse all nodes in the tree using BFS and 
        // keep only nodes at height > maxHeight - CUT_OFF_AGE
        List<Node> candidates = new LinkedList<>();
        Queue<Node> queue = new LinkedBlockingQueue<>();
        queue.add(chain);
        while (!queue.isEmpty()) {
            Node p = queue.remove();
            if (p.height >= maxHeight - CUT_OFF_AGE) {
                candidates.add(p);
                updateUTXPool(p.block);
                // System.out.println("\t" + p.block + ", h = " + p.height);
            }
            for (Node node: p.children)
                queue.add(node);
        }
        // System.out.println("Number of candidates = " + candidates.size());
        // for (Node node: candidates)
        //     System.out.println("\t" + node.block);
        if (candidates.isEmpty())
            candidates.add(chain);
        Node parent = null;
        ByteArrayWrapper w = new ByteArrayWrapper(block.getPrevBlockHash());
        for (Node node: candidates) {
            if (new ByteArrayWrapper(node.block.getHash()).equals(w)) {
                parent = node;
                break;
            }
        }
        if (parent == null)
            return false;
        // check validity of the block's transactions 
        List<Transaction> txs = block.getTransactions();
        UTXOPool uPool = uPoolMap.get(parent.block);
        TxHandler handler = new TxHandler(uPool);
        Transaction[] rTxs = handler.handleTxs(txs.toArray(new Transaction[txs.size()]));
        if (rTxs.length != txs.size()) {
            // System.out.println("\tThere exists invalid transactions: " + txs.size() + " != " + rTxs.length);
            return false;
        }
        // update the pool of this block
        uPoolMap.put(block, handler.getUTXOPool());

        Node current = new Node(block, parent);
        current.height = parent.height + 1;
        parent.children.add(current);

        // updateUTXPool(block);

        // removed all transactions of this block from the transaction pool
        for (Transaction tx: block.getTransactions())
            txPool.removeTransaction(tx.getHash());

        // remove all utxo claimed by the parent
        // UTXOPool pool = uPoolMap.get(parent.block);
        // for (UTXO utxo: pool.getAllUTXO())
        //     pool.removeUTXO(utxo);

        // update the genesis block as the first (oldest) candidate
        // chain = candidates.get(0);
        // System.out.println("current txPool = " + txPool.getTransactions());
        return true;
    }

    /** Add a transaction to the transaction pool */
    public void addTransaction(Transaction tx) {
        // IMPLEMENT THIS
        txPool.addTransaction(tx);
    }
}