from ScraperFC import Transfermarkt
from ScraperFC import FBRef
import pandas as pd
import boto3
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import time
AWS_KEY = '...'
AWS_SECRET_KEY = '...'
s3 = boto3.client('s3',aws_access_key_id=AWS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY)
BUCKET ='ds-scraper-stats'

def read_df(website,league,year,name):
    obj_s3 = s3.get_object(Bucket = BUCKET, Key = f'{website}/{league}/{year}/{name}.csv')
    df= pd.read_csv(obj_s3['Body'])
    return df


def tm_create_df(leagues,years):
    df = pd.DataFrame(columns=['name_tm','team_tm','link_tm','league_tm','year_tm'])
    for league in leagues:
        for year in years:
            df_league_year = read_df("Transfermarkt",league,year,'players')
            df_league_year = df_league_year.rename(columns={'Player':'name_tm','Squad':'team_tm','Player_link':'link_tm'})
            df_league_year = df_league_year[['name_tm','team_tm','link_tm']]
            df_league_year ['league_tm'] = league
            df_league_year ['year_tm'] = int(year)
            df = pd.concat([df,df_league_year]).reset_index(drop=True)
    return df

def fbref_create_df(leagues, years):
    df = pd.DataFrame(columns=['name_fbref','team_fbref','link_fbref','league_fbref','year_fbref'])
    for league in leagues:
        for year in years:
            df_league_year = read_df("FBref-processed-opta",league,year,'player_stats')
            df_league_year = df_league_year.rename(columns={'Player':'name_fbref','Squad':'team_fbref','Player Link':'link_fbref'})
            df_league_year = df_league_year[['name_fbref','team_fbref','link_fbref']]
            df_league_year ['league_fbref'] = league
            df_league_year ['year_fbref'] = int(year)
            df = pd.concat([df,df_league_year]).reset_index(drop=True)
    return df

