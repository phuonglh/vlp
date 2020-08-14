package vlp.ner

/**
 * @author Phuong LE-HONG, phuonglh@gmail.com
 * <p>
 * Sep 6, 2016, 10:46:20 AM
 * <p>
 * Different types of annotation for each token of a sentence which 
 * are defined as enumeration values.
 *
 */
object Label extends Enumeration {
  val PartOfSpeech, Chunk, NamedEntity, RegexpType = Value 
}