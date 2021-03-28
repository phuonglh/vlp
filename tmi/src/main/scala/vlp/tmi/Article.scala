package vlp.tmi

/**
  * GDELT article structure.
  * 
  * phuonglh@gmail.com
  *
  * @param url
  * @param url_mobile
  * @param title
  * @param seendate
  * @param socialimage
  * @param domain
  * @param language
  * @param sourcecountry
  */
case class Article(
    url: String,
    url_mobile: String,
    title: String,
    seendate: String,
    socialimage: String,
    domain: String,
    language: String,
    sourcecountry: String
)

case class Document(url: String, content: String)