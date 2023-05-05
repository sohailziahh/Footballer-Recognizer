
import requests
from bs4 import BeautifulSoup
import time
HEADERS = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '+\
                    'AppleWebKit/537.36 (KHTML, like Gecko) '+\
                    'Chrome/55.0.2883.87 Safari/537.36'
            }

def transfermarkt_scraper(url_plyr):
    # url_plyr = self.df['link_tm']        
    response = requests.get(url_plyr, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    time.sleep(0.1)
    # try :
    ### total mins played so far in the league 
    # caps =soup.find_all("div", {"class": "box viewport-tracking"})
    # if caps[2].find("div", {"id":"player-performance-table"}):
    #     href = "https://www.transfermarkt.us" + caps[2].find("a")["href"]
    #     soup_href = BeautifulSoup(requests.get(href, headers=self.headers).content, 'html.parser')
    #     total= soup_href.find("table", {"class": "items"}).find('tfoot').find_all("td",{"class":"rechts"})[-1]
    #     total = re.findall(r'\d+', str(total))
    #     if len(total)==0:
    #         total_min = 0
    #     else:
    #         total_min = int(''.join(total)) 
    # else:
    #     total_min = int(-1)
    try:
        link_image = soup.find("div", {"class":"modal__content"}).find("img")["src"]
    except:
        link_image = ''

    return link_image

def fbref_scraper(url_plyr):
    s = requests.Session()
    try :
        content = s.get(url_plyr, timeout= 30)
        soup = BeautifulSoup(content.content, 'lxml')
        link_image = soup.find('div',{'class': 'media-item'}).find("img").get('src')  
    except:
        link_image= ''
    return link_image