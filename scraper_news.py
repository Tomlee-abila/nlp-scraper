import requests
from bs4 import BeautifulSoup
import pandas as pd
import uuid
from datetime import datetime
import time
import re

def scrape_aljazeera():
    base_url = "https://www.aljazeera.com"
    sitemap_url = "https://www.aljazeera.com/sitemap.xml"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        response = requests.get(sitemap_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'xml')
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

    # Find recent post sitemaps (e.g., usually post-sitemap.xml or sitemap-article.xml)
    # Al Jazeera sitemaps are year-month based
    sitemap_locs = [loc.text for loc in soup.find_all('loc') if 'sitemap.xml' in loc.text]
    
    # If no nested sitemaps found, maybe it's the article sitemap itself
    if not sitemap_locs:
        sitemap_locs = [sitemap_url]

    article_urls = []
    
    for sm_loc in reversed(sitemap_locs): # start from most recent
        print(f"Fetching URLs from sub-sitemap: {sm_loc}")
        try:
            sm_resp = requests.get(sm_loc, headers=headers, timeout=15)
            sm_soup = BeautifulSoup(sm_resp.content, 'xml')
            locs = [loc.text for loc in sm_soup.find_all('loc')]
            for l in locs:
                if '/news/' in l or '/economy/' in l:
                    article_urls.append(l)
            
            if len(article_urls) >= 350:
                break
        except Exception as e:
            print(f"Error fetching {sm_loc}: {e}")

    # Remove duplicates and limit
    article_urls = list(set(article_urls))[:350]
    
    articles_data = []
    count = 1
    
    for url in article_urls:
        if len(articles_data) >= 300:
            break
            
        print(f"\n{count}. scraping {url}")
        print("        requesting ...")
        
        try:
            art_resp = requests.get(url, headers=headers, timeout=15)
            if art_resp.status_code != 200:
                print("        failed (bad status)")
                continue
                
            print("        parsing ...")
            art_soup = BeautifulSoup(art_resp.content, 'html.parser')
            
            headline_tag = art_soup.find('h1')
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            
            # Find body paragraphs -> usually <p> tags inside the article content
            # Al Jazeera usually puts content in wysiwyg main-content
            paragraphs = art_soup.find_all('p')
            body_text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
            if len(body_text) < 100:
                continue # skip pseudo-articles or videos
                
            # Date scraping (approx)
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_tag = art_soup.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                date_str = time_tag['datetime'][:10]
                
            articles_data.append({
                'uuid': str(uuid.uuid4()),
                'URL': url,
                'date': date_str,
                'headline': headline,
                'body': body_text
            })
            
            path_msg = "data/articles.csv"
            print(f"        saved in {path_msg}")
            count += 1
            time.sleep(0.5) # respect rate limits
            
        except Exception as e:
            print(f"        failed to parse {url}: {e}")

    return articles_data

if __name__ == "__main__":
    print("Starting scraper...")
    data = scrape_aljazeera()
    
    if len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv("data/articles.csv", index=False)
        print(f"\nSuccessfully scraped {len(data)} articles.")
    else:
        print("\nNo articles scraped.")
