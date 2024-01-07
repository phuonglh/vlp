package vlp.dep

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * A sentence is simply a mutable list of tokens.
  * @param tokens a list buffer of tokens
  */
case class Sentence(tokens: ListBuffer[Token]) {
  private val undefined = Token("UNK", mutable.Map[Label.Value, String]())
  
  def length: Int = tokens.length

  /**
    * Gets a slice of this sentence between a pair of indices.
    * @param startIndex start index, inclusive
    * @param endIndex end index, exclusive
    * @return a portion of the sentence
    */
  def slice(startIndex: Int, endIndex: Int): Sentence = {
    Sentence(tokens.slice(startIndex, endIndex))
  }

  /**
    * Searches for a token with a given id, if that token does not exist then 
    * returns the ROOT token.
    * @param tokenId the id of the token
    */
  def token(tokenId: String): Token = {
    tokens.find(t => t.id == tokenId).getOrElse(undefined)
  }
}
