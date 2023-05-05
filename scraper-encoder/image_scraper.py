from unidecode import unidecode
import urllib.request
from url_links import tm_create_df,fbref_create_df
from scrapers import transfermarkt_scraper, fbref_scraper
from encoders import encode_deefpface,encode_facerecognition,check_for_headhsot
from tqdm import tqdm
import pandas as pd
import os

def image_save(df,link_image,website):
        if not os.path.exists(f"headhost_{website}"):
            os.makedirs(f"headhost_{website}")
        try :
            path = f"./headhost_{website}/{df['name']}_{df[f'league_{website}']}_{unidecode(df[f'team_{website}'])}.jpg"
            urllib.request.urlretrieve(link_image, path)
        except:
            path = ''
        return path
class scraper:
    def __init__(self,encode=True) -> None:
        self.leagues = ['EPL', 'La Liga']#, 'Bundesliga', 'Serie A', 'Ligue 1']
        self.year = [2023]
        self.available_websites = ["fbref","transfermarkt"]
        self.websites_abv = {'fbref':'fbref','transfermarkt':'tm'}
        self.encode = encode
    def prepare_df(self):
        for v in list(self.websites_abv.values()): 
            self.df[[f"path_{v}"]] =''
            if self.encode:
                self.df[[f"encoding_{v}"]] =''
    def image_scraper(self,websites):
        self.websites = websites
        check = self.check_wbesits()
        if check:
            self.df = self.get_combine_rescources()
            self.prepare_df()
            for index, row in tqdm(self.df.iterrows()):
                link_tm = self.df.iloc[index]['link_tm']
                link_fbref = self.df.iloc[index]['link_fbref']
                if pd.isnull(link_tm) == False:
                    if "transfermarkt" in self.websites:
                        row['path_tm'] = image_save(row,transfermarkt_scraper(link_tm),self.websites_abv['transfermarkt'])  
                    if "fbref" in self.websites:
                        row['path_fbref'] = image_save(row,fbref_scraper(link_fbref),self.websites_abv['fbref'])  
                    if self.encode:  
                        row['encoding_tm'] = check_for_headhsot(encode_facerecognition(row['path_tm']))
                        row['encoding_fbref'] = check_for_headhsot(encode_facerecognition(row['path_fbref']))
            return self.df      
        else:
            print("The input website(s) are not scrapable")
            return -1

    def check_wbesits(self):
        for website in self.websites:
            if website in self.available_websites:
                return True
            else:
                return False

    def get_combine_rescources(self):
        _df_tm = tm_create_df(self.leagues,self.year)
        # _df_tm.sort_values(by=['name_tm']).to_csv("tm.csv")    
        _df_fbref = fbref_create_df(self.leagues,self.year)
        # _df_fbref.sort_values(by=['name_fbref']).to_csv("fbref.csv")
        if len(self.websites)>=2:
            _df_fbref['name'] = [unidecode(f).lower() for f in _df_fbref['name_fbref']]
            _df_fbref = _df_fbref.drop_duplicates(subset='link_fbref')
            _df_tm['name'] = [unidecode(t).lower() for t in _df_tm['name_tm']]
            _df_tm = _df_tm.drop_duplicates(subset='link_tm')
            _df_fbref = _df_fbref.set_index('name')
            _df_tm = _df_tm.set_index('name')
            df_all = pd.merge(_df_fbref, _df_tm,  left_index=True, right_index=True).reset_index()
        return df_all

   
    


