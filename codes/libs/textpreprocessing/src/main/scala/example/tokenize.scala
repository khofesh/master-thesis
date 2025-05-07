package tokenizetext

import scala.io.Source
import com.github.tototoshi.csv._
import java.io._
import collection.JavaConverters._
//InaNLP
import IndonesianNLP._

object tokenizeText extends App{
  
  val input = "out1.csv"
  val tokenizer = new IndonesianSentenceTokenizer
  val output = new File("out2.csv")
  
  for(line <- Source.fromFile(input).getLines){
    var token = tokenizer.tokenizeSentenceWithCompositeWords(line)
    val writer = CSVWriter.open(output, append=true)
    writer.writeRow(List(token))
    writer.close()
    
  }
}
