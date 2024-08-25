
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from requests.exceptions import RequestException
from langchain.tools import tool
from fake_useragent import UserAgent
import pandas as pd
import re
import os
import asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
import chromadb

load_dotenv()
OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')


embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def get_random_user_agent():
    ua = UserAgent()
    return ua.random

def extract_url(web_str):
    # Use regular expression to find the URL
    url_match = re.search(r'http[s]?://\S+', web_str)
    if url_match:
        return url_match.group()
    else:
        return None

def scrape_new_website(url: str, base_domain: str, max_retries: int = 3, backoff_factor: float = 0.3, timeout: int = 10) -> dict:
    headers = {'User-Agent': get_random_user_agent()}
    session = requests.Session()
    # Extract the URL
    url = extract_url(url)
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Basic content cleaning
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = 'joined '.join(chunk for chunk in chunks if chunk)
            
            # Extract links within the same domain
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            sublinks = [link for link in links if urlparse(link).netloc == base_domain]
            
            print(len(sublinks))
            return {
                "source": url,
                "content": text,
                "links": sublinks[:2]
            }
        
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                return {
                    "source": url,
                    "error": f"Failed to scrape website after {max_retries} attempts: {str(e)}"
                }
            else:
                time.sleep(backoff_factor * (2 ** attempt))
                continue

def get_links_and_text_new(url: str, max_depth: int = 2, max_retries: int = 3, backoff_factor: float = 0.3, timeout: int = 10):
    visited_urls = set()
    results = []

    def scrape_recursive(url: str, depth: int):
        if depth > max_depth or url in visited_urls:
            return

        visited_urls.add(url)
        base_domain = urlparse(url).netloc
        result = scrape_new_website(url, base_domain, max_retries, backoff_factor, timeout)
        
        if "error" not in result:
            results.append({"source": result["source"], "content": result["content"]})
            for link in result.get("links", []):
                scrape_recursive(link, depth + 1)

    scrape_recursive(url, 0)
    return results




