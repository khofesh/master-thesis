package processtext2

import scala.io.Source
import com.github.tototoshi.csv._
import java.io._
import collection.JavaConverters._
//InaNLP
import IndonesianNLP._

object processText2 extends App{
  
  val filename = "input.csv"
  val tokenizer = new IndonesianSentenceTokenizer
  val formalizer = new IndonesianSentenceFormalization
  val stemmer = new IndonesianStemmer
  val f = new File("output.csv")
  
  for(line <- Source.fromFile(filename).getLines){
    // remove dot with space
    var sentence = line.replaceAll("\\.", " ")
    // replace comma with space
    sentence = sentence.replaceAll("\\,", " ")
    
    // normalize words and eliminate stopwords
    sentence = formalizer.formalizeSentence(sentence)
    // delete stop words
    formalizer.initStopword
    sentence = formalizer.deleteStopword(sentence)
    // remove punctuation
    sentence = sentence.replaceAll("""([\p{Punct}&&[^.]]|\b\p{IsLetter}{1,2}\b)\s*""", "")
  
    // stem sentence
    sentence = stemmer.stemSentence(sentence)
    
    // write to csv file
    val bw = new BufferedWriter( new FileWriter(f, true))
    bw.write(sentence + "\n")
    bw.close()
  
  }
}
