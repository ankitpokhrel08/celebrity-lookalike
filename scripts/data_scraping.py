import os
import requests
import time
import urllib.parse
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
import hashlib
import random

class PlayerImageScraper:
    def __init__(self, base_folder="players/images"):
        self.base_folder = base_folder
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup Chrome options for better scraping
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        
    def create_player_folder(self, player_name):
        """Create a folder for the player if it doesn't exist"""
        # Clean player name for folder creation
        folder_name = player_name.lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        folder_path = os.path.join(self.base_folder, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        
        return folder_path, folder_name
    
    def get_google_image_urls(self, search_query, max_images=15):
        """Scrape Google Images for image URLs using the specific format"""
        # Setup Chrome driver with webdriver-manager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        
        try:
            # Create the Google Images search URL with the specific format
            # Using the exact format from your example: "player_name cricketer single photos"
            encoded_query = urllib.parse.quote(search_query)
            search_url = f"https://www.google.com/search?q={encoded_query}&udm=2&sxsrf=AE3TifMgahni9Ls-NXtYbgR3xdDy0owB_w&biw=1470&bih=798&dpr=2"
            
            print(f"Searching: {search_url}")
            driver.get(search_url)
            
            # Wait for images to load
            time.sleep(3)
            
            # Scroll to load more images
            for i in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                # Try to click "Show more results" if available
                try:
                    show_more = driver.find_element(By.XPATH, "//input[@value='Show more results']")
                    driver.execute_script("arguments[0].click();", show_more)
                    time.sleep(3)
                except:
                    pass
            
            # Find all image elements
            image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
            print(f"Found {len(image_elements)} image elements")
            
            image_urls = []
            for img in image_elements:
                try:
                    # Get the image URL from different possible attributes
                    img_url = None
                    
                    # Try different attributes where the image URL might be stored
                    for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
                        url = img.get_attribute(attr)
                        if url and url.startswith('http') and 'base64' not in url:
                            img_url = url
                            break
                    
                    if img_url:
                        # Filter out unwanted URLs (like icons, logos, etc.)
                        if any(skip in img_url.lower() for skip in ['logo', 'icon', 'button', 'arrow', 'googlelogo']):
                            continue
                            
                        # Filter out very small images by checking URL patterns
                        if any(small in img_url for small in ['=s16', '=s32', '=s48', '=w16', '=w32', '=w48', '=h16', '=h32', '=h48']):
                            continue
                        
                        # Modify URL to get larger image if possible
                        if '=s' in img_url or '=w' in img_url or '=h' in img_url:
                            # Replace small size parameters with larger ones
                            img_url = re.sub(r'=s\d+', '=s400', img_url)
                            img_url = re.sub(r'=w\d+', '=w400', img_url)
                            img_url = re.sub(r'=h\d+', '=h400', img_url)
                        
                        if img_url not in image_urls:
                            image_urls.append(img_url)
                            print(f"Found image URL: {img_url[:100]}...")
                            
                            if len(image_urls) >= max_images:
                                break
                                
                except Exception as e:
                    continue
            
            # If we didn't find enough images, try clicking on images to get high-res versions
            if len(image_urls) < max_images:
                print("Trying to get high-resolution images by clicking...")
                try:
                    clickable_images = driver.find_elements(By.CSS_SELECTOR, "div[data-ri] img, a img")
                    for i, img in enumerate(clickable_images[:10]):
                        try:
                            driver.execute_script("arguments[0].click();", img)
                            time.sleep(2)
                            
                            # Look for the high-res image that appears
                            high_res_imgs = driver.find_elements(By.CSS_SELECTOR, "img[src*='http']")
                            for hr_img in high_res_imgs:
                                hr_url = hr_img.get_attribute('src')
                                if (hr_url and hr_url.startswith('http') and 
                                    hr_url not in image_urls and 
                                    'base64' not in hr_url and
                                    len(hr_url) > 50):  # Longer URLs usually mean higher res
                                    image_urls.append(hr_url)
                                    print(f"Found high-res image: {hr_url[:100]}...")
                                    break
                            
                            if len(image_urls) >= max_images:
                                break
                                
                        except Exception as e:
                            continue
                except Exception as e:
                    print(f"Error getting high-res images: {e}")
            
            print(f"Total URLs found: {len(image_urls)}")
            return image_urls[:max_images]
            
        except Exception as e:
            print(f"Error scraping Google Images: {e}")
            return []
        finally:
            driver.quit()
    
    def download_image(self, url, folder_path, index):
        """Download an image from URL and save it to the specified folder"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://www.google.com/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
            }
            
            response = self.session.get(url, headers=headers, timeout=15, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if not any(img_type in content_type.lower() for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
                return False
            
            # Create filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            file_extension = self.get_file_extension(content_type, url)
            filename = f"{index:02d}_{url_hash}{file_extension}"
            
            file_path = os.path.join(folder_path, filename)
            
            # Don't download if file already exists
            if os.path.exists(file_path):
                return True
            
            # Download and save
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify the file was downloaded and has reasonable content
            if os.path.getsize(file_path) > 2048:  # At least 2KB
                print(f"Downloaded: {filename} ({os.path.getsize(file_path)} bytes)")
                return True
            else:
                # Remove small/corrupted files
                os.remove(file_path)
                return False
                
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return False
    
    def get_file_extension(self, content_type, url):
        """Get appropriate file extension"""
        if 'jpeg' in content_type or 'jpg' in content_type:
            return '.jpg'
        elif 'png' in content_type:
            return '.png'
        elif 'webp' in content_type:
            return '.webp'
        elif 'gif' in content_type:
            return '.gif'
        else:
            # Try to get from URL
            if '.png' in url.lower():
                return '.png'
            elif '.webp' in url.lower():
                return '.webp'
            elif '.gif' in url.lower():
                return '.gif'
            else:
                return '.jpg'
    
    def scrape_player_images(self, player_name, max_images=10):
        """Scrape and download images for a single player"""
        print(f"\nScraping images for: {player_name}")
        
        # Create player folder
        folder_path, folder_name = self.create_player_folder(player_name)
        
        # Check existing images
        existing_images = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))]
        
        if len(existing_images) >= max_images:
            print(f"Player {player_name} already has {len(existing_images)} images. Skipping...")
            return
        
        needed_images = max_images - len(existing_images)
        print(f"Need {needed_images} more images for {player_name}")
        
        # Create the exact search query format you specified
        search_query = f"{player_name} cricketer mugshot"
        print(f"Search query: {search_query}")
        
        # Get image URLs from Google Images
        image_urls = self.get_google_image_urls(search_query, max_images=max_images * 2)  # Get more than needed
        
        if not image_urls:
            print(f"No images found for {player_name}")
            return
        
        print(f"Found {len(image_urls)} image URLs for {player_name}")
        
        # Download images
        downloaded_count = len(existing_images)
        for i, url in enumerate(image_urls):
            if downloaded_count >= max_images:
                break
            
            print(f"Downloading image {downloaded_count + 1}/{max_images} for {player_name}")
            
            if self.download_image(url, folder_path, downloaded_count + 1):
                downloaded_count += 1
            
            # Add delay between downloads to be respectful
            time.sleep(random.uniform(1, 3))
        
        total_images = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))])
        print(f"Successfully downloaded {downloaded_count - len(existing_images)} new images")
        print(f"Total images for {player_name}: {total_images}")

def read_players_from_file(file_path="players.txt"):
    """Read player names from the text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            players = [line.strip() for line in f.readlines() if line.strip()]
        return players
    except FileNotFoundError:
        print(f"Players file {file_path} not found!")
        return []

def main():
    print("Cricket Player Image Scraper")
    print("=" * 50)
    
    # Read player names
    players = read_players_from_file("players.txt")
    
    if not players:
        print("No players found in players.txt file!")
        return
    
    print(f"Found {len(players)} players to scrape images for")
    
    # Initialize scraper
    scraper = PlayerImageScraper()
    
    # Create base directory
    if not os.path.exists(scraper.base_folder):
        os.makedirs(scraper.base_folder)
    
    # Scrape images for each player
    for i, player in enumerate(players, 1):
        print(f"\n{'='*60}")
        print(f"Progress: {i}/{len(players)} - {player}")
        print(f"{'='*60}")
        
        try:
            scraper.scrape_player_images(player, max_images=20)
        except Exception as e:
            print(f"Error processing {player}: {e}")
        
        # Add delay between players to avoid being blocked
        if i < len(players):
            wait_time = random.uniform(5, 10)
            print(f"Waiting {wait_time:.1f} seconds before next player...")
            time.sleep(wait_time)
    
    print("\n" + "="*60)
    print("Image scraping completed!")
    print("="*60)

if __name__ == "__main__":
    main()
