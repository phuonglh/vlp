package vlp.tok

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths, StandardOpenOption}

import vlp.VLP

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.control.Breaks._
import scala.util.matching.Regex

case class Brick(name: String, pattern: Regex, weight: Int = 0)

/**
  * Scala implementation of a Vietnamese tokenizer.
  *
  * June 2019, phuonglh@gmail.com
  *
  */
object Tokenizer {
  /**
    * A list of bricks which capture common token types.
    */
  val bricks = List[Brick](
    Brick("code", raw"""\b\d+\p{Lu}+\b""".r, 5),
    Brick("road", raw"""\b([Qq]uốc lộ|[Tt]ỉnh lộ|[Hh]uyện lộ|[Đđ]ường)\s+(\d+\p{Lu}+|\p{Lu}+\d+)\b""".r, 5),
    Brick("docNo", raw"""\b([Ll]uật|[Nn]ghị quyết|[Cc]ông văn|[Qq]uyết định|[Nn]ghị định|[Tt]hông tư|[Tt]ờ trình|[Tt]hông báo|[Cc]hỉ thị|[Vv]ăn bản)(\s+(số))?\s+(\d+[\p{Lu}-]*/)+[\p{Lu}-]*\b""".r, 5),
    Brick("acronym", raw"""\b\p{Lu}\p{Ll}\.?\p{Lu}+\.?\b""".r, 5),
    Brick("email", raw"""(\w[-._%:\w]*@\w[-._\w]*\w\.\w{2,3})""".r, 4),
    Brick("url", raw"""(((\w+)\://)+[a-zA-z][\-\w]*\w+(\.\w[\-\w]*)+(/[\w\-]+)*(\.\w+)?(/?)(\?(\w+=[\w%]+))*(&(\w+=[\w%]+))*|[a-z]+((\.)\w+)+)""".r, 3),
    Brick("name", raw"""\b(\p{Lu}\p{Ll}*)([\s+_&\-]?(\p{Lu}\p{Ll}*))+\b""".r, 2),
    Brick("allCapital", raw"""\b[\p{Lu}]{2,}([\s_][\p{Lu}]{2,})*\b""".r, 1),
    Brick("date1", raw"""\b(([12][0-9]|3[01]|0*[1-9])[-/.](1[012]|0*[1-9])[-/.](\d{4}|\d{2})|(1[012]|0*[1-9])[-/.]([12][0-9]|3[01]|0*[1-9])[-/.](\d{4}|\d{2}))\b""".r, 1),
    Brick("date2", raw"""\b(1[012]|0*[1-9])[-/](\d{4}|\d{2})\b""".r, 1),
    Brick("date3", raw"""\b([12][0-9]|3[01]|0*[1-9])[-/](1[012]|0*[1-9])\b""".r, 1),
    Brick("date4", raw"""\b([Nn]gày)(\s+)\d+(\s+)tháng(\s+)\d+(\s+)năm(\s+)(\d+)\b""".r, 1),
    Brick("time", raw"""\b\d{1,2}:\d{1,2}\b""".r, 1),
    Brick("numberSeq", raw"""\+?\d+(([\s.-]+\d+)){2,}\b""".r, 1),
    Brick("duration", raw"""\b\d{4}\-\d{4}\b""".r, 1),
    Brick("currency", raw"""\p{Sc}+\s?(\d*)?\d+([.,]\d+)*\b""".r),
    Brick("number", raw"""([+-]?(\d*)?[\d]+([.,]\d+)*%?)""".r),
    Brick("item", raw"""\d+[.)]\b""".r),
    Brick("bracket", raw"""[\}\{\]\[><\)\(]+""".r),
    Brick("capital", raw"""\b\p{Lu}+[\p{Ll}_-]*[+]?\b""".r),
    Brick("phrase", raw"""\b[\p{Ll}\s_\-]+\b""".r),
    Brick("punct", raw"""[-@…–~`'“”’‘|\/.,:;!?'\u0022]+""".r),
    Brick("other", raw""".+""".r, -1)
  )

  /**
    * The Vietnamese lexicon
    */
  val lexicon = Source.fromInputStream(Tokenizer.getClass.getResourceAsStream("/lexicon.txt"), "UTF-8").getLines().toSet


  def segment(syllables: Array[String], forward: Boolean = false, verbose: Boolean = false): List[String] = {
    if (forward) segmentForward(syllables, verbose); else segmentBackward(syllables, verbose)
  }


  /**
    * Maximum matching from left to right.
    * @param syllables
    * @param verbose
    * @return a list of words
    */
  def segmentForward(syllables: Array[String], verbose: Boolean): List[String] = {
    val result = ListBuffer[String]()
    if (syllables.size >= 1) {
      var token = syllables.head
      var word = token
      val n = Math.min(11, syllables.size - 1)
      var m = 0
      for (i <- 0 until n) {
        token = token + ' ' + syllables(i+1)
        if (lexicon.contains(token)) {
          word = token
          m = i + 1
          if (verbose) println(word)
        }
      }
      result += word
      if (m < n) {
        val right = segmentForward(syllables.slice(m+1, syllables.size), verbose)
        result ++= right
      }
    }
    result.toList
  }

  /**
    * Maximum matching from right to left.
    * @param syllables
    * @param verbose
    * @return a list of words
    */
  def segmentBackward(syllables: Array[String], verbose: Boolean): List[String] = {
    val result = ListBuffer[String]()
    if (syllables.size >= 1) {
      var token = syllables.last
      var word = token
      val n = Math.max(0, syllables.size - 11)
      var m = syllables.size - 1
      for (i <- (m-1) to n by -1) {
        token = syllables(i) + ' ' + token
        if (lexicon.contains(token)) {
          word = token
          m = i
          if (verbose) println(word)
        }
      }
      result += word
      if (m > 0) {
        val left = segmentBackward(syllables.slice(0, m), verbose)
        result ++= left
      }
    }
    result.toList
  }

  def run(input: String, forward: Boolean = false, verbose: Boolean = false): List[(Int, String, String)] = {
    val text = input.replaceAll("\\u0022", " ")
    val result = ListBuffer[(Int, String, String)]()
    if (text.trim.nonEmpty) {
      breakable {
        for (brick <- bricks) {
          val m = brick.pattern.findFirstMatchIn(text)
          if (m != None) {
            val x = (m.get.start, brick.name, m.get.matched)
            if (verbose) println(s"\t$x")
            if (x._1 > 0) {
              val left = text.substring(0, x._1).trim
              if (verbose) println(s"left = ${left}")
              val ys = run(left, forward, verbose)
              result ++= ys
            }
            if (x._2 == "phrase") {
              val syllables = x._3.split("\\s+")
              var tokens = segment(syllables, forward, verbose)
              if (!forward) tokens = tokens.reverse
              for (token <- tokens) result += ((x._1, "word", token.trim))
            } else result += x
            if (x._1 < text.size) {
              val right = text.substring(x._1 + x._3.size).trim
              if (verbose) println(s"right = ${right}")
              val ys = run(right, forward, verbose)
              result ++= ys
            }
            break // do not consider the next brick
          }
        }
      }
    }
    result.toList
  }

  /**
    * Merges capital tokens with their subsequent word tokens if necessary.
    * ["Quản", "lý"] => ["Quản lý"].
    */
  def merge(tokens: List[(Int, String, String)], verbose: Boolean = false): List[(Int, String, String)] = {
    if (verbose && tokens.size > 256) {
      VLP.log(s"Merging ${tokens.size} tokens...")
    }
    val result = ListBuffer[(Int, String, String)]()
    var i = 0
    val n = tokens.size
    if (n >= 2) {
      while (i < n && tokens(i)._2 != "capital") i = i + 1
      for (j <- 0 until i) result += tokens(j)
      if (i < n) {
        var token = tokens(i)._3
        var w = token
        var k = i
        for (j <- i+1 until Math.min(i+3, n)) {
          w = w + ' ' + tokens(j)._3
          if (lexicon.contains(w.toLowerCase)) {
            k = j
            token = w
          }
        }
        result += ((i, "capital", token))
        val rest = merge(tokens.slice(k+1, n), verbose)
        result ++= rest
      }
    } else if (n >= 1) result += tokens.head
    result.toList
  }

  /**
    * Splits and merges two tokens [name, lower], for example ["Bộ Quốc", "phòng"] => ["Bộ", "Quốc phòng"].
    * @param tokens
    * @param verbose
    */
  def split(tokens: List[(Int, String, String)], verbose: Boolean = false): List[(Int, String, String)] = {
    val result = ListBuffer[(Int, String, String)]()
    var i = 0
    val n = tokens.size
    if (n >= 2) {
      while (i < n && tokens(i)._2 != "name") i = i + 1
      for (j <- 0 until i) result += tokens(j)
      var nextIndex = i+1
      if (i < n-1) {
        val token = tokens(i)._3
        val ss = token.split("\\s+")
        val w = ss.last + " " + tokens(i+1)._3
        if (lexicon.contains(w.toLowerCase)) {
          result += ((i, "capital", ss.slice(0, ss.size-1).mkString(" ")))
          result += ((i + 1, "capital", w))
          nextIndex = i+2
        } else result += tokens(i)
        if (nextIndex < n) {
          val rest = split(tokens.slice(nextIndex, n), verbose)
          result ++= rest
        }
      } else if (i < n) result += tokens(i)
    } else if (n >= 1) result += tokens.head
    result.toList
  }

  /**
    * Tokenizes a text into tokens; each token is a tuple (index, shape, word).
    * @param text a text, usually a sentence
    * @param forward use forward maximal matching instead of backward.
    * @param verbose
    * @return a list of tokens
    */
  def tokenize(text: String, forward: Boolean = false, verbose: Boolean = false): List[(Int, String, String)] = {
    val tokens = split(merge(run(text, forward, verbose)))
    val nonSpaceTokens = tokens.filter(t => t._3.trim.size > 0 && t._3.trim != "\u00A0")
    // re-index the tokens, merge syllables with underscore and return result
    nonSpaceTokens.zipWithIndex.map(pair => (pair._2 + 1, pair._1._2, pair._1._3.replaceAll("\\s+", "_")))
  }

  def tokenizeOne(text: String): List[List[(Int, String, String)]] = {
    val sentence = Unicode.convert(text)
    val backwardTokens = tokenize(sentence)
    val forwardTokens = tokenize(sentence, true)
    val different = backwardTokens.zip(forwardTokens).exists(a => a._1._3 != a._2._3)
    if (different) List(backwardTokens, forwardTokens); else List(backwardTokens)
  }

  /**
    * Tokenizes a sequence of texts into tokens; each token is a tuple (index, shape, word).
    * @param texts a sequence of texts
    * @param parallel
    *
    * */
  def tokenizeMany(texts: Seq[String], parallel: Boolean = false): List[List[(Int, String, String)]] = {
    val result = if (parallel) {
      texts.par.map(text => tokenize(Unicode.convert(text)))
    } else {
      texts.map(text => tokenize(Unicode.convert(text)))
    }
    result.toList
  }

  /**
    * Tokenizes a input text file and write the result to an output text file or to the console.
    * @param inputPath
    * @param outputPath
    * @param parallel
    */
  def process(inputPath: String, outputPath: String = "", parallel: Boolean = false): Unit = {
    VLP.log("Reading input file...")
    val texts = Source.fromFile(inputPath).getLines().filter(_.trim.nonEmpty).toList
    VLP.log(s"#(sentences) = ${texts.size}")
    VLP.log("Tokenizing the sentences... Please wait.")
    val start = System.currentTimeMillis()
    val output = tokenizeMany(texts, parallel)
    val end = System.currentTimeMillis()
    VLP.log(s"Duration = ${(end - start) / 1000.0} (seconds)")
    if (outputPath.trim().nonEmpty) {
      val sentences = output.map(s => s.map(_._3).mkString(" "))
      import collection.JavaConversions._
      Files.write(Paths.get(outputPath), sentences, StandardCharsets.UTF_8, StandardOpenOption.CREATE)
      println(s"Result was written into the output file ${outputPath}")
    } else {
      output.foreach(s => VLP.log(s.mkString(" ")))
    }
  }

  def main(args: Array[String]): Unit = {
    VLP.log(s"#(words) = ${lexicon.size}")
    if (args.size >= 2) {
      process(args(0), args(1), true)
    } else if (args.size >= 1) 
        process(args(0))
    else {
      val text = """Tiếp nhận công văn số 10209/SKHĐT-PTHT ngày 14/11/2016 của Sở Kế hoạch và Đầu tư Tôi đến Sở Tài nguyên và Môi trường để gặp anh Nguyễn Văn Nam."""
      val result = tokenizeOne(text)
      result.foreach(a => VLP.log(a.mkString(" ")))
    }
    VLP.log("Done.")
  }
}
