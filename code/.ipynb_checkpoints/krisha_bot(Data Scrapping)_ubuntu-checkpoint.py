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
    def __init__(self, output_dir="krisha_data", dataset_file="krisha_dataset.json", csv_output_path=None):
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
        self.csv_output_path = csv_output_path  # Added custom CSV output path
        
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
        """Scrape data from a single apartment listing page."""
        try:
            print(f"Scraping listing: {url}")
            
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Check for rate limiting or captcha
            if "captcha" in response.text.lower() or response.status_code == 429:
                print("Rate limiting or captcha detected. Waiting before retrying...")
                time.sleep(random.uniform(30, 60))  # Wait longer before retry
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract listing ID from URL
            listing_id_match = re.search(r'/a/show/(\d+)', url)
            if listing_id_match:
                listing_id = listing_id_match.group(1)
            else:
                listing_id = f"unlabeled_{int(time.time())}"
            
            # Initialize listing data dictionary
            listing_data = {
                'id': listing_id,
                'url': url
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
            
            # Extract contact information (only name, not phone)
            try:
                contact_name_elem = soup.select_one("div.owners__name")
                listing_data['contact_name'] = contact_name_elem.text.strip() if contact_name_elem else "Unknown"
            except Exception as e:
                print(f"Error extracting contact name: {e}")
                listing_data['contact_name'] = "Unknown"
            
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
            
            # Save the data to our dataset
            self.dataset[listing_id] = listing_data
            
            # Save the dataset after each listing to avoid losing data
            self.save_dataset()
            
            print(f"Finished scraping listing {listing_id}.")
            return True
        except Exception as e:
            print(f"Error scraping listing {url}: {e}")
            return False
    
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
                
                # Check for rate limiting or captcha
                if "captcha" in response.text.lower() or response.status_code == 429:
                    print("Rate limiting or captcha detected. Waiting before retrying...")
                    time.sleep(random.uniform(30, 60))  # Wait longer before retry
                    continue
                
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
                    success = self.scrape_listing_page(listing_url)
                    if success:
                        total_scraped += 1
                    
                    # Add a delay between processing listings
                    time.sleep(random.uniform(1.0, 3.0))
                
                # Add a longer delay between pages
                time.sleep(random.uniform(3.0, 5.0))
                
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                break
        
        print(f"Scraping completed. Processed {total_listings} listings and successfully scraped {total_scraped} listings.")
        return total_listings, total_scraped
    
    def export_to_csv(self, output_file=None):
        """Export the dataset to a CSV file for analysis."""
        import csv
        
        if not self.dataset:
            print("No data to export.")
            return
        
        # If custom CSV path is provided, use it
        if self.csv_output_path:
            output_file = self.csv_output_path
        # Otherwise, if no specific output file is provided, use city name if available
        elif output_file is None:
            if self.city:
                output_file = f"krisha_{self.city}_dataset.csv"
            else:
                output_file = "krisha_dataset.csv"
        
        # Collect all possible fields from all listings
        fields = set()
        for listing_data in self.dataset.values():
            fields.update(listing_data.keys())
        
        # Sort fields for consistent output
        fields = sorted(list(fields))
        
        # Ensure key fields are at the beginning
        for key_field in ['id', 'city', 'title', 'rooms', 'area_sqm', 'floor', 'district', 'price', 'contact_name']:
            if key_field in fields:
                fields.remove(key_field)
                fields.insert(0, key_field)
        
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for listing_id, listing_data in self.dataset.items():
                    # Create a copy of the data for CSV export
                    export_data = listing_data.copy()
                    writer.writerow(export_data)
            
            print(f"Dataset exported to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None


def run_krisha_scraper(url, output_dir=None, max_pages=1, dataset_file=None, csv_output_path=None):
    """Function to run the Krisha scraper with direct parameters"""
    # Clean up the URL to ensure it's in the right format
    # Remove trailing slashes and query parameters for consistency
    url = url.split('?')[0].rstrip('/')
    
    # Extract city from URL for default naming
    city_match = re.search(r'/kvartiry/([^/]+)', url)
    city = city_match.group(1) if city_match else "unknown"
    
    # Set default output directory based on city
    if output_dir is None:
        output_dir = f"krisha_{city}_data"
    
    # Set default dataset file based on city
    if dataset_file is None:
        dataset_file = f"krisha_{city}_dataset.json"
    
    # Initialize scraper with appropriate settings
    scraper = KrishaScraper(output_dir=output_dir, dataset_file=dataset_file, csv_output_path=csv_output_path)
    
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

# Example usage with custom CSV path:
# result = run_krisha_scraper(
#     "https://krisha.kz/arenda/kvartiry/almaty/", 
#     output_dir="almaty_rentals", 
#     max_pages=1,
#     csv_output_path="/path/to/your/custom/location/almaty_data.csv"
# )
# print(f"Total listings collected: {len(result['dataset'])}")

# Replace the original example at the bottom of the script with this one that uses csv_output_path
# dataset = run_krisha_scraper("https://krisha.kz/arenda/kvartiry/almaty/", output_dir="almaty_rentals", max_pages=1)


result = run_krisha_scraper(
                             "https://krisha.kz/arenda/kvartiry/almaty/",
                              output_dir="almaty_rentals",
                              max_pages = 3,
                              csv_output_path="/root/krisha_bot/data/regular_scrapping/"
)