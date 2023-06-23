import urllib.request

webUrl = urllib.request.urlopen('https://www.pearson.it/letteraturapuntoit/contents/files/manz_sintesi.pdf')
print (webUrl.read())
