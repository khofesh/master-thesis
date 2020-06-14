# 
https://www.tokopedia.com/souvigameshop/paling-murah-new-google-chromecast-2-2015?trkid=f=Ca27L000P0W0S0Sh00Co0Po0Fr0Cb0_src=directory_page=1_ob=5_q=_catid=665_po=1

selProdPage = Selector(text=driver.page_source)

# harga
price = selProd.xpath('//div[@class="product-box-content"]/descendant::span[@itemprop="price"]/text()').extract()[0].replace('.', '')

# merchant (biasa/gold merchant)
merchant = []
if selProdPage.xpath('//div[@class="product-box-content"]/descendant::div/i/@data-original-title').extract() is None:
    merchant = ['biasa']
else:
    merchant = ['gold']

# nama merchant
merchantName = selProdPage.xpath('//div[@class="product-box-content"]/descendant::a[@id="shop-name-info"]/text()').extract()

# reputasi merchant
reputasi = selProdPage.xpath('//div[@class="product-box-content"]/descendant::img/@data-original-title').extract()[0].replace(' points', '').replace('.', '')

# produk dilihat
produkDilihat = selProdPage.xpath('//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "view-count")]/text()').extract()[0]

# produk tersebut dibeli
produkTerjual = selProdPage.xpath('//div[contains(@class, "product-content-container")]/descendant::dd[contains(@class, "item-sold-count")]/text()').extract()

# nilai produk (ratings/nilai produk (1-5), jumlah ulasan)
# buka halaman "Ulasan"
driver.find_element_by_xpath('//li[@id="p-nav-review"]').click()
#simpan page source
selProdPage2 = Selector(text=driver.page_source)
# nilai produk
nilaiProduk = selProdPage2.xpath('//div[contains(@class, "reviewsummary-loop")]/div/p/text()').extract()
# jumlah ulasan
# hasil extract() terdapat \xa0 (non-breaking space in Latin1 (ISO 8859-1))
jumlahUlasan = selProdPage2.xpath('//div[contains(@class, "reviewsummary-loop")]/descendant::div[@class="mt-5"]/text()').extract()[0].replace('\xa0', ' ').split(' ')[0]
# jumlah bintang 5
jml5star = selProdPage2.xpath('//div[contains(@class, "ratingtotal")]/text()').extract()[0].replace(' ','').replace('\n', '')
# jumlah bintang 4
jml4star = selProdPage2.xpath('//div[contains(@class, "ratingtotal")]/text()').extract()[1].replace(' ','').replace('\n', '')
# jumlah bintang 3
jml3star = selProdPage2.xpath('//div[contains(@class, "ratingtotal")]/text()').extract()[2].replace(' ','').replace('\n', '')
# jumlah bintang 2
jml2star = selProdPage2.xpath('//div[contains(@class, "ratingtotal")]/text()').extract()[3].replace(' ','').replace('\n', '')
# jumlah bintang 1
jml1star = selProdPage2.xpath('//div[contains(@class, "ratingtotal")]/text()').extract()[4].replace(' ','').replace('\n', '')

## Ulasan paling membantu
# gambar reviewers
imgReviewers = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::img[@class="list-box-image"]/@src').extract()
# rating yg diberikan reviewers
# outputnya:
# ['rating-star5 most__helpful__rating-star mb-10 mt-5',
# 'rating-star5 most__helpful__rating-star mb-10 mt-5',
# 'rating-star5 most__helpful__rating-star mb-10 mt-5']
ratingGvnByReviewer = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/span/@class').extract()
for i in range(len(ratingGvnByReviewers)):
    ratingGvnByReviewers[i] = ratingGvnByReviewers[i].split(' ')[0].replace('rating-star', '')
# komen yang diberikan
commentGvnByReviewer = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "most__helpful")]/div[@class="relative"]/p/text()').extract()
# positif, negatif, netral
smileyPositifReviewer = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "green")]/text()').extract()
smileyNetralReviewer = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "yellorange")]/text()').extract()
smileyNegatifReviewer = selProdPage2.xpath('//div[contains(@class, "multiple-mosthelpful")]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "red")]/text()').extract()


for img in imgReviewers:
     for nama in namaReviewers:
         for status in statusReviewers:
             for rating in ratingGvnByReviewers:
                 for comment in commentGvnByReviewers:
                     yield {"jenisUlasan": "palingMembantu",
                     "imgReviewers": img,
                     "namaReviewers": nama,
                     "statusReviewers": status,
                     "ratingGvnByReviewers": rating,
                     "commentGvnByReviewers": comment
                     }

## Ulasan biasa
# gambar reviewer
imgReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::img[@class="list-box-image"]/@src').extract()
# rating
ratingGvnByReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/div/i/@class').extract()
# output:
# ['rating-star rating-star3',
# 'rating-star rating-star5',
# 'rating-star rating-star5',
# 'rating-star rating-star1',
# 'rating-star rating-star5',
# 'rating-star rating-star5',
# 'rating-star rating-star5',
# 'rating-star rating-star5',
# 'rating-star rating-star5',
# 'rating-star rating-star5']
for i in range(len(ratingGvnByReviewerBiasa)):
    ratingGvnByReviewerBiasa[i] = ratingGvnByReviewerBiasa[i].replace('rating-star rating-star', '')
# komen yang diberikan 
commentGvnByReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[contains(@class, "list-box-text")]/span[@class="review-body"]/text()').extract()
# Number of people who find reviews helpful
reviewHelpful = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[@class="like-review"]/div/text()').extract()
for i in range(len(reviewHelpful)):
    reviewHelpful[i] = reviewHelpful[i].replace(' orang lainnya terbantu dengan ulasan ini', '')
# smiley
smileyPositifReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "green")]/text()').extract()
smileyNetralReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "yellorange")]/text()').extract()
smileyNegatifReviewerBiasa = selProdPage2.xpath('//ul[@id="review-container"]/descendant::div[contains(@class, "smile-tooltip-hover")]/descendant::span[contains(@class, "red")]/text()').extract()
  

# click ">"
driver.find_element_by_xpath('//div[@class="pagination"]/descendant::i[@class="icon-chevron-right"]').click()

# setelah seluruh review di extract
# click "Diskusi Produk)
driver.find_element_by_xpath('//li[@id="p-nav-talk"]').click()
# jumlah diskusi
jumlahDiskusi = selProdPage2.xpath('//li[@id="p-nav-talk"]/a/span/text()').extract()

## kategori dan subkategori belom

# URL ketika diurut berdasarkan ulasan
# page 1
https://www.tokopedia.com/p/elektronik/media-player?ob=5
# page 2
https://www.tokopedia.com/p/elektronik/media-player?ob=5&page=2
# page 3
https://www.tokopedia.com/p/elektronik/media-player?ob=5&page=3

prodKategori = selPage.xpath('//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()').extract()[0]
prodSubKategori = selPage.xpath('//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()').extract()[1]
prodSubSubKategori = selPage.xpath('//div[@id="breadcrumb-container"]/descendant::li/h2/a/text()').extract()[2]
prodNama = selPage.xpath('//div[@id="breadcrumb-container"]/descendant::li/h2/text()').extract()

## struktur database mongodb
##
from pymongo import MongoClient
client = MongoClient()
db = client.producturi

collection = db.prodpage
result = collection.insert_one(
    {
        "uri":"https://www.tokopedia.com/tokotoped/authentic-ud-kanthal-a1-wire-26-awg-04mm-vaporizer-vapor-rda?trkid=f=Ca2090L000P0W0S0Sh00Co0Po0Fr0Cb0_src=directory_page=1_ob=5_q=_po=13_catid=2094",
        "topads": "no",
        "IDurlproductDB": "28302349",
        "prodName": "Authentic UD Kanthal A1 Wire 26 AWG | 0.4mm | vaporizer vapor rda",
        "prodPrice": "1424",
        "prodCat": "Elektronik ",
        "prodSubCat": "Vaporizer ",
        "prodSubSubCat": "Coil ",
        "merchantName": "Toko Toped",
        "merchantType": "gold",
        "merchantRep": merchantRep,
        "prodSeen": prodSeen,
        "prodSold": prodSold,
        "prodPic": prodPic,
        "prodRating": prodRating,
        "reviewCount": reviewCount,
        "cnt5star": cnt5star,
        "cnt4star": cnt4star,
        "cnt3star": cnt3star,
        "cnt2star": cnt2star,
        "cnt1star": cnt1star,
        "helpfulReviewers": [
            {
                "imgReviewers": something,
                "ratingGvnByReviewers": something,
                "commentGvnByReviewers": something,
                "posSmileyReviewers": something,
                "neutSmileyReviewers": something,
                "negSmileyReviewers": something
            },
            {
                "imgReviewers": something,
                "ratingGvnByReviewers": something,
                "commentGvnByReviewers": something,
                "posSmileyReviewers": something,
                "neutSmileyReviewers": something,
                "negSmileyReviewers": something
            }
        ],
        "ordinaryReviewers": [
            {
                "imgReviewersOrd": something,
                "ratingGvnByReviewersOrd": something,
                "commentGvnByReviewersOrd": something,
                "reviewOrdHelpful": something,
                "posSmileyReviewersOrd": something,
                "neutSmileyReviewersOrd": something,
                "negSmileyReviewersOrd": something
            },
            {
                "imgReviewersOrd": something,
                "ratingGvnByReviewersOrd": something,
                "commentGvnByReviewersOrd": something,
                "reviewOrdHelpful": something,
                "posSmileyReviewersOrd": something,
                "neutSmileyReviewersOrd": something,
                "negSmileyReviewersOrd": something
            }
        ]
    })

# hasil dari insert_one()
# yaitu ObjectId dan dapat dilihat dengan
# inserted_id
result.inserted_id

# adding new ordinary reviewers or helpful reviewers
# into the array:
# $addToSet only add the value if the value does 
# not already exist in the array
db.customers.update(
    {"_id": result.inserted_id},
    {"$addToSet": {"helpfulReviewers":
        {"bla": bla,
        "bla": bla
        }
     }
    }
)



