


# Veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Burada amacımız verilen puanları tarihe göre
# ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılmasını sağlayacağım.


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("dataset/amazon_review.csv")
df.head()
df.columns


df["overall"].mean()
#Ürünün ortalama puanını 4.587 olarak buldum.



df["reviewTime"].dtypes

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

current_date = df["reviewTime"].max()

df["days"] = (current_date - df["reviewTime"]).dt.days

df.loc[df["days"] <= 30, "overall"].mean() * 30 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] < 90), "overall"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] < 180), "overall"].mean() * 23 / 100 + \
df.loc[df["days"] > 180, "overall"].mean() * 21 / 100

#reviewTime değişkenini tarih değişkenine dönüştürdüm. reviewTime 'ın max değerini current date
#olarak atadım. puan-yorum tarihi ile current date'in farkını alarak yeni bir days değişkeni oluşturdum.
#günleri ağırlıklandırarak ürün ortalamasını hesapladım.

#ortalama değerim 4.705 olarak geldi.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

#toplam oy sayısından(total_vote) yararlı bulunan oy sayısını (helpful_yes) çıkararak helpful_no isimli yararlı bulunmayan oy sayısı
#değişkeni türettim.

def score_pos_neg_diff(up, down):
    return up - down


df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])
df.head()


def score_average_rating(up, down):
    if (up + down) == 0:
        return 0
    return (up / (up + down))


df["score_average_rating"] =df.apply(lambda x:score_average_rating(x["helpful_yes"], x["helpful_no"]),axis=1)
df.head()

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x : wilson_lower_bound(x["helpful_yes"], x["helpful_no"]),axis=1)


#score_pos_neg_diff, score_average_rating, wilson_lower_bound 'a göre skorlar oluşturarak bu değerleri aynı
#isimli değişkenlere atadım.


df.sort_values("wilson_lower_bound", ascending=False).head(20)

#wilson lower bound'a göre büyükten küçüğe doğru skorları sıralayarak ilk 20 yorumu gözlemlemiş oldum.















