# -*- coding: utf-8 -*-
"""twitter-scraping-username.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kaVDowacJdkRfYy8ExS7xkctX2DeT2oC
"""

import os
import pandas as pd

print('Lutfen pip3 install snscrape komutunu kullanarak kullanarak kodun çalışması için gerekli kütüphaneyi kurunuz')

tweet_count = int(input('Kac adet tweet cekmek istiyorsunuz giriniz : '))
username = str(input('Lütfen tweet çekmek istediğiniz kullanıcı adını giriniz :'))

# Using OS library to call CLI commands in Python
os.system("snscrape --jsonl --max-results {} twitter-search 'from:{}'> user-tweets.json".format(tweet_count, username))
tweets_df1 = pd.read_json('user-tweets.json', lines=True)

tweets_df1.to_csv('user-tweets.csv', sep=',', index=False)

print('İslem basariyla tamamlanmistir')
