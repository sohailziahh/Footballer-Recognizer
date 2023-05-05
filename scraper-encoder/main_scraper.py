from image_scraper import scraper
import pickle

scraper = scraper()
df = scraper.image_scraper(["fbref","transfermarkt"])
# f = open('headshots_encoding.pickle', "wb")
# f.write(pickle.dumps(df))