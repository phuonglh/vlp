package blc;

import java.security.PublicKey;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
  * phuonglh@gmail.com
  * </p>
  * Copied from the txs package.
  * 
  */

public class TxHandler {
    private UTXOPool pool;

    /**
     * Creates a public ledger whose current UTXOPool (collection of unspent transaction outputs) is
     * {@code utxoPool}. This should make a copy of utxoPool by using the UTXOPool(UTXOPool uPool)
     * constructor.
     */
    public TxHandler(UTXOPool utxoPool) {
        // IMPLEMENT THIS: DONE
        pool = new UTXOPool(utxoPool);
    }

    /**
     * @return true if:
     * (1) all outputs claimed by {@code tx} are in the current UTXO pool, 
     * (2) the signatures on each input of {@code tx} are valid, 
     * (3) no UTXO is claimed multiple times by {@code tx},
     * (4) all of {@code tx}s output values are non-negative, and
     * (5) the sum of {@code tx}s input values is greater than or equal to the sum of its output
     *     values; and false otherwise.
     */
    public boolean isValidTx(Transaction tx) {
        // IMPLEMENT THIS: DONE
        ArrayList<Transaction.Input> inputs = tx.getInputs();
        ArrayList<Transaction.Output> outputs = tx.getOutputs();
        // (1) outputs claimed here mean inputs to this transaction
        for (int j = 0; j < inputs.size(); j++) {
            UTXO utxo = new UTXO(inputs.get(j).prevTxHash, inputs.get(j).outputIndex);
            if (!pool.contains(utxo)) 
                return false;
        }
        // (2)
        double totalInputValue = 0d;
        for (int j = 0; j < inputs.size(); j++) {
            byte[] message = tx.getRawDataToSign(j);
            byte[] signature = inputs.get(j).signature;
            UTXO previousUtxo = new UTXO(inputs.get(j).prevTxHash, inputs.get(j).outputIndex);
            Transaction.Output output = pool.getTxOutput(previousUtxo);
            if (output != null)
                totalInputValue += output.value;
            PublicKey pubKey = output.address;
            if (!Crypto.verifySignature(pubKey, message, signature))
                return false;
        }
        // (3)
        Set<UTXO> uxs = new HashSet<>();
        for (Transaction.Input input: inputs) {
            UTXO utxo = new UTXO(input.prevTxHash, input.outputIndex);
            uxs.add(utxo);
        }
        if (uxs.size() < inputs.size()) // duplicated UTXOs detected by object comparison
            return false;
        // (4) and (5)
        double totalOutputValue = 0d;
        for (int j = 0; j < outputs.size(); j++) {
            Transaction.Output output = outputs.get(j);
            if (output.value < 0)
                return false;
            totalOutputValue += output.value;
        }
        if (totalInputValue < totalOutputValue)
            return false;
        return true;
    }

    /**
     * Handles each epoch by receiving an unordered array of proposed transactions, checking each
     * transaction for correctness, returning a mutually valid array of accepted transactions, and
     * updating the current UTXO pool as appropriate.
     */
    public Transaction[] handleTxs(Transaction[] possibleTxs) {
        // IMPLEMENT THIS
        Set<Transaction> txs = new HashSet<>();
        for (Transaction tx: possibleTxs) {
            if (isValidTx(tx)) {
                txs.add(tx);
                // update pool by adding more unspent transaction outputs
                List<Transaction.Output> outputs = tx.getOutputs();
                for (int j = 0; j < outputs.size(); j++) {
                    UTXO utxo = new UTXO(tx.getHash(), j);
                    pool.addUTXO(utxo, outputs.get(j));
                }
            } 
        }
        return txs.toArray(new Transaction[txs.size()]);
    }

    // added by phuonglh for assignment 3 as guided
    public UTXOPool getUTXOPool() {
        return pool;
    }
}
