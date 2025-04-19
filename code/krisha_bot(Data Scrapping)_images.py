import requests
from bs4 import BeautifulSoup
import os
import time
import re
import random
import json
from urllib.parse import urljoin
import pandas as pd

class KrishaScraper:
    def __init__(self, output_dir="krisha_images", dataset_file="krisha_dataset.json"):
        """Initialize the scraper with default settings."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8,kk;q=0.7",
            "Connection": "keep-alive",
            "Referer": "https://www.google.com/",
        }
        self.session = requests.Session()
        self.base_url = "https://krisha.kz"
        self.output_dir = output_dir
        self.dataset_file = dataset_file
        self.city = None  # Will be set based on the URL
        self.dataset = self.load_dataset()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_dataset(self):
        """Load existing dataset if available."""
        if os.path.exists(self.dataset_file):
            try:
                with open(self.dataset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print(f"Error loading dataset file. Creating new dataset.")
                return {}
        return {}
    
    def save_dataset(self):
        """Save dataset to JSON file."""
        with open(self.dataset_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {self.dataset_file}")
    
    def download_image(self, url, file_name):
        """Download an image and save it to the specified filename."""
        try:
            response = self.session.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {file_name}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_numeric_id(self, listing_id):
        """Extract numeric ID from listing_id string."""
        if isinstance(listing_id, str) and 'listing_' in listing_id:
            return listing_id.replace('listing_', '')
        return listing_id
    
    def parse_title(self, title):
        """Parse title to extract key apartment information."""
        data = {}
        
        # Example title: "2-комнатная квартира, 60 м², 3/9 этаж, Бостандыкский р-н — Навои за 280 000 〒"
        if title and title != "Unknown":
            # Extract rooms
            rooms_match = re.search(r'(\d+)-комнатная', title)
            if rooms_match:
                data['rooms'] = rooms_match.group(1)
            
            # Extract area
            area_match = re.search(r'(\d+(?:\.\d+)?)\s*м²', title)
            if area_match:
                data['area_sqm'] = area_match.group(1)
            
            # Extract floor
            floor_match = re.search(r'(\d+)/(\d+)\s*этаж', title)
            if floor_match:
                data['floor'] = floor_match.group(1)
                data['total_floors'] = floor_match.group(2)
            
            # Extract district
            district_match = re.search(r'(\w+(?:-\w+)*)\s+р-н', title)
            if district_match:
                data['district'] = district_match.group(1)
            
            # Extract address/location after district
            address_match = re.search(r'р-н\s+—\s+(.+?)\s+(?:за|аренда)', title)
            if address_match:
                data['address'] = address_match.group(1).strip()
        
        return data
    
    def extract_price(self, price_text):
        """Extract price value from price text."""
        if price_text and price_text != "Unknown":
            # Remove spaces and non-breaking spaces
            clean_price = price_text.replace(' ', '').replace('\xa0', '')
            
            # Extract numeric part
            price_match = re.search(r'(\d+)', clean_price)
            if price_match:
                return price_match.group(1)
        
        return "Unknown"
    
    def parse_parameters(self, params_dict):
        """Parse parameters into structured data."""
        structured_data = {}
        
        # Map Russian/Kazakh parameter names to English keys
        param_mapping = {
            'Город': 'city',
            'Район': 'district',
            'Жилой комплекс': 'residential_complex',
            'Квартира': 'apartment_type',
            'Планировка': 'layout',
            'Площадь': 'area',
            'Площадь кухни': 'kitchen_area',
            'Этаж': 'floor',
            'Ремонт': 'renovation',
            'Санузел': 'bathroom',
            'Балкон': 'balcony',
            'Мебель': 'furniture',
            'Год постройки': 'year_built',
            'Парковка': 'parking',
            'Отопление': 'heating',
            'Безопасность': 'security',
            'Разное': 'miscellaneous'
        }
        
        for key, value in params_dict.items():
            if key in param_mapping:
                eng_key = param_mapping[key]
                structured_data[eng_key] = value
        
        return structured_data
    
    def scrape_listing_page(self, url):
        """Scrape images and data from a single apartment listing page."""
        try:
            print(f"Scraping listing: {url}")
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract listing ID from URL
            listing_id_match = re.search(r'/a/show/(\d+)', url)
            if listing_id_match:
                listing_id = listing_id_match.group(1)
            else:
                listing_id = f"unlabeled_{int(time.time())}"
            
            # Create directory for this listing
            image_dir = os.path.join(self.output_dir, f"listing_{listing_id}")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            
            # Initialize listing data dictionary
            listing_data = {
                'id': listing_id,
                'url': url,
                'images': []
            }
            
            # If city is known from URL, add it to listing data
            if self.city:
                listing_data['city'] = self.city
            
            # Find the apartment title
            try:
                title_elem = soup.select_one("div.offer__advert-title h1")
                title = title_elem.text.strip() if title_elem else "Unknown"
                listing_data['title'] = title
                
                # Parse the title to extract apartment details
                title_data = self.parse_title(title)
                listing_data.update(title_data)
            except:
                listing_data['title'] = "Unknown"
            
            # Extract price
            try:
                price_elem = soup.select_one("div.offer__price")
                price_text = price_elem.text.strip() if price_elem else "Unknown"
                listing_data['price_raw'] = price_text
                
                # Parse price to extract value
                listing_data['price'] = self.extract_price(price_text)
            except:
                listing_data['price_raw'] = "Unknown"
                listing_data['price'] = "Unknown"
            
            # Extract contact information
            try:
                contact_name_elem = soup.select_one("div.owners__name")
                listing_data['contact_name'] = contact_name_elem.text.strip() if contact_name_elem else "Unknown"
                
                # Extract the phone number if available
                phone_button = soup.select_one("button.show-phones")
                if phone_button and 'data-offer-id' in phone_button.attrs:
                    listing_data['has_phone'] = True
                else:
                    listing_data['has_phone'] = False
            except:
                listing_data['contact_name'] = "Unknown"
                listing_data['has_phone'] = False
            
            # Extract description
            try:
                description_elem = soup.select_one("div.offer__description")
                if description_elem:
                    listing_data['description'] = description_elem.text.strip()
            except:
                listing_data['description'] = "No description"
            
            # Extract parameters
            params_dict = {}
            try:
                params = soup.select("div.offer__parameters dl")
                for param in params:
                    dt = param.select_one("dt")
                    dd = param.select_one("dd")
                    if dt and dd:
                        key = dt.text.strip()
                        value = dd.text.strip()
                        params_dict[key] = value
                
                # Parse parameters into structured data
                params_data = self.parse_parameters(params_dict)
                listing_data.update(params_data)
                
                # Keep the original parameters dictionary
                listing_data['parameters_raw'] = params_dict
            except Exception as e:
                print(f"Error extracting parameters: {e}")
                listing_data['parameters_raw'] = {}
            
            # Find location/address details
            try:
                address_elem = soup.select_one("div.offer__location")
                if address_elem:
                    listing_data['full_address'] = address_elem.text.strip()
            except:
                pass
            
            # Find and download full-sized images
            image_links = []
            
            # Method 1: Try to find gallery thumbnails with data-href attribute
            thumbs = soup.select("button.gallery__thumb-image.js__gallery-thumb")
            for thumb in thumbs:
                if 'data-href' in thumb.attrs:
                    image_url = thumb['data-href']
                    if not image_url.startswith('http'):
                        image_url = urljoin("https://krisha-photos.kcdn.online/", image_url)
                    # Only collect JPG images, skip WEBP
                    if image_url.lower().endswith('.jpg'):
                        image_links.append(image_url)
            
            # If we didn't find any JPG images using method 1, try other methods
            if not image_links:
                # Method 2: Look for picture elements with source tags
                picture_elements = soup.select("picture source")
                for source in picture_elements:
                    if 'srcset' in source.attrs:
                        srcset = source['srcset']
                        # Extract jpg URLs only
                        jpg_urls = re.findall(r'(https?://[^\s]+\.jpg[^\s]*)', srcset)
                        if jpg_urls:
                            # Get the last URL which is typically the highest resolution
                            full_img_url = jpg_urls[-1].split(' ')[0]
                            # Convert to full-size image URL by replacing dimensions
                            full_img_url = re.sub(r'-\d+x\d+\.jpg', r'-full.jpg', full_img_url)
                            if full_img_url not in image_links:
                                image_links.append(full_img_url)
                
                # Method 3: Look for image elements
                if not image_links:
                    img_elements = soup.select("img.gallery__main-image")
                    for img in img_elements:
                        if 'src' in img.attrs:
                            img_url = img['src']
                            if not img_url.startswith('http'):
                                img_url = urljoin(self.base_url, img_url)
                            # Only collect JPG images, skip WEBP
                            if img_url.lower().endswith('.jpg'):
                                if img_url not in image_links:
                                    image_links.append(img_url)
            
            # Filter out potential duplicates based on URL path
            unique_image_links = []
            seen_images = set()
            
            for img_url in image_links:
                # Extract the base filename without extension to identify duplicate images
                base_name = re.search(r'/([^/]+?)(?:-\w+)?\.(?:jpg|webp)', img_url.lower())
                if base_name:
                    img_id = base_name.group(1)
                    if img_id not in seen_images:
                        seen_images.add(img_id)
                        unique_image_links.append(img_url)
                else:
                    # If we can't extract a meaningful ID, just add the URL
                    unique_image_links.append(img_url)
            
            # Download found images
            downloaded_count = 0
            for i, img_url in enumerate(unique_image_links, 1):
                # Ensure we're only downloading JPG images
                if not img_url.lower().endswith('.jpg'):
                    continue
                    
                file_name = f"image_{i:02d}.jpg"
                file_path = os.path.join(image_dir, file_name)
                if self.download_image(img_url, file_path):
                    downloaded_count += 1
                    listing_data['images'].append({
                        'index': i,
                        'url': img_url,
                        'local_path': os.path.join(f"listing_{listing_id}", file_name)
                    })
                
                # Add a small delay to avoid overloading the server
                time.sleep(random.uniform(0.5, 1.5))
            
            # Add number of images to the data
            listing_data['images_count'] = len(unique_image_links)
            listing_data['images_downloaded'] = downloaded_count
            
            # Save the data to our dataset
            self.dataset[listing_id] = listing_data
            
            # Save the dataset after each listing to avoid losing data
            self.save_dataset()
            
            print(f"Finished scraping listing {listing_id}. Downloaded {downloaded_count} JPG images.")
            return downloaded_count
        except Exception as e:
            print(f"Error scraping listing {url}: {e}")
            return 0
    
    def extract_city_from_url(self, url):
        """Extract city name from search URL."""
        # Example: https://krisha.kz/arenda/kvartiry/almaty/
        match = re.search(r'/kvartiry/([^/]+)', url)
        if match:
            return match.group(1)
        return None
    
    def scrape_search_results(self, search_url, max_pages=1):
        """Scrape multiple listings from search results pages."""
        total_scraped = 0
        total_listings = 0
        
        # Extract city from the URL
        self.city = self.extract_city_from_url(search_url)
        if self.city:
            print(f"Detected city: {self.city}")
        
        # First, visit the main page to establish cookies
        try:
            print("Visiting main page to establish session...")
            main_response = self.session.get(self.base_url, headers=self.headers)
            main_response.raise_for_status()
            time.sleep(random.uniform(1.0, 2.0))
        except Exception as e:
            print(f"Error accessing main page: {e}")
        
        # Make sure search_url doesn't end with '?page=1'
        search_url = search_url.rstrip('/')
        if search_url.endswith('?page=1'):
            search_url = search_url[:-7]
        
        for page in range(1, max_pages + 1):
            # Construct the correct URL for each page
            # For first page, use the base URL without any page parameter
            if page == 1:
                page_url = search_url + '/'
            else:
                # For subsequent pages, add the page parameter
                page_url = f"{search_url}/?page={page}"
            
            try:
                print(f"Scraping search results page {page}: {page_url}")
                
                response = self.session.get(page_url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all listing links - specifically for Krisha apartment listings
                listings = soup.select("div.a-card__inc a.a-card__title")
                
                if not listings:
                    # Try alternative selector
                    listings = soup.select("a.a-card__title")
                
                if not listings:
                    print(f"No listings found on page {page}. Stopping.")
                    
                    # Save the HTML for debugging
                    debug_file = f"debug_page_{page}.html"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"Saved page HTML to {debug_file} for debugging")
                    break
                
                print(f"Found {len(listings)} listings on page {page}")
                total_listings += len(listings)
                
                for i, listing in enumerate(listings):
                    listing_url = urljoin(self.base_url, listing['href'])
                    print(f"Processing listing {i+1}/{len(listings)} on page {page}")
                    download_count = self.scrape_listing_page(listing_url)
                    total_scraped += download_count
                    
                    # Add a delay between processing listings
                    time.sleep(random.uniform(1.0, 3.0))
                
                # Add a longer delay between pages
                time.sleep(random.uniform(3.0, 5.0))
                
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                break
        
        print(f"Scraping completed. Processed {total_listings} listings and downloaded {total_scraped} images.")
        return total_listings, total_scraped
    
    def export_to_csv(self, output_file=None):
        """Export the dataset to a CSV file for analysis."""
        import csv
        
        if not self.dataset:
            print("No data to export.")
            return
        
        # If no specific output file is provided, use city name if available
        if output_file is None:
            if self.city:
                output_file = f"krisha_{self.city}_dataset.csv"
            else:
                output_file = "krisha_dataset.csv"
        
        # Collect all possible fields from all listings
        fields = set()
        for listing_data in self.dataset.values():
            fields.update(listing_data.keys())
        
        # Remove the 'images' field as it's a list and handle it separately
        if 'images' in fields:
            fields.remove('images')
        
        # Sort fields for consistent output
        fields = sorted(list(fields))
        
        # Ensure key fields are at the beginning
        for key_field in ['id', 'city', 'title', 'rooms', 'area_sqm', 'floor', 'district', 'price']:
            if key_field in fields:
                fields.remove(key_field)
                fields.insert(0, key_field)
        
        # Add image count field
        fields.append('image_paths')
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for listing_id, listing_data in self.dataset.items():
                    # Create a copy of the data for CSV export
                    export_data = {k: v for k, v in listing_data.items() if k != 'images'}
                    
                    # Add image paths as semicolon-separated string
                    if 'images' in listing_data and listing_data['images']:
                        image_paths = ';'.join([img['local_path'] for img in listing_data['images']])
                        export_data['image_paths'] = image_paths
                    
                    writer.writerow(export_data)
            
            print(f"Dataset exported to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None


def run_krisha_scraper(url, output_dir=None, max_pages=1, dataset_file=None):
    """Function to run the Krisha scraper with direct parameters"""
    # Clean up the URL to ensure it's in the right format
    # Remove trailing slashes and query parameters for consistency
    url = url.split('?')[0].rstrip('/')
    
    # Extract city from URL for default naming
    city_match = re.search(r'/kvartiry/([^/]+)', url)
    city = city_match.group(1) if city_match else "unknown"
    
    # Set default output directory based on city
    if output_dir is None:
        output_dir = f"krisha_{city}_images"
    
    # Set default dataset file based on city
    if dataset_file is None:
        dataset_file = f"krisha_{city}_dataset.json"
    
    # Initialize scraper with appropriate settings
    scraper = KrishaScraper(output_dir=output_dir, dataset_file=dataset_file)
    
    if "/a/show/" in url:
        # Single listing
        scraper.scrape_listing_page(url)
    else:
        # Search results page
        results = scraper.scrape_search_results(url, max_pages=max_pages)
    
    # Export the dataset to CSV for easy analysis
    csv_file = scraper.export_to_csv()
    
    return {
        "dataset": scraper.dataset,
        "dataset_file": dataset_file,
        "csv_file": csv_file,
        "output_dir": output_dir
    }

# Example usage:
# result = run_krisha_scraper("https://krisha.kz/arenda/kvartiry/almaty/", max_pages=2)
# print(f"Total listings collected: {len(result['dataset'])}")

dataset = run_krisha_scraper("https://krisha.kz/arenda/kvartiry/almaty/", output_dir="almaty_rentals", max_pages=500)


