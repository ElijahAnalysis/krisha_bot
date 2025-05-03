import os
import re
import random
import pandas as pd
import numpy as np
import logging
import joblib
import time
import threading
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, ConversationHandler, filters

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Path configurations
DATA_PATH = "/root/krisha_bot/data/regular_scrapping/cleaned/almaty_apartments_clustered.csv"
PRICE_MODEL_PATH = "/root/krisha_bot/models/krisha_almaty_rental_stacking.joblib"
IMAGES_DIR = "/root/krisha_bot/data/regular_scrapping/images/almaty_rental_images/images"

# Mapping dictionaries
BATHROOM_MAPPING = {
    0: '2 с/у и более', 
    1: 'неизвестно', 
    2: 'разделен', 
    3: 'разделен, совмещен', 
    4: 'раздельный', 
    5: 'совмещен', 
    6: 'совмещенный'
}

BATHROOM_ENCODING = {
    '2 с/у и более': 0,
    'разделен': 2,
    'совмещен': 5
}

PARKING_MAPPING = {
    0: 'гараж', 
    1: 'неизвестно', 
    2: 'паркинг', 
    3: 'рядом охраняемая стоянка'
}

DISTRICT_MAPPING = {
    0: 'Алматы',
    1: 'Алматы, Алатауский р-н',
    2: 'Алматы, Алмалинский р-н',
    3: 'Алматы, Ауэзовский р-н',
    4: 'Алматы, Бостандыкский р-н',
    5: 'Алматы, Жетысуский р-н',
    6: 'Алматы, Медеуский р-н',
    7: 'Алматы, Наурызбайский р-н',
    8: 'Алматы, Турксибский р-н'
}

DISTRICT_CHOICES = {
    'Алмалинский': 2,
    'Бостандыкский': 4,
    'Наурызбайский': 7,
    'Весь Алматы': 0,
    'Ауэзовский': 3,
    'Медеуский': 6,
    'Алатауский': 1,
    'Турксибский': 8,
    'Жетысуский': 5
}

# Conversation states
MAIN_MENU, DISTRICT_SELECTION, VIEWING_LISTINGS, PRICE_ESTIMATION_FLOOR, PRICE_ESTIMATION_TOTAL_FLOORS, PRICE_ESTIMATION_AREA, PRICE_ESTIMATION_ROOMS, PRICE_ESTIMATION_BATHROOM = range(8)

class KrishaBot:
    def __init__(self):
        # Load data and model
        self.df = pd.read_csv(DATA_PATH)
        self.price_model = joblib.load(PRICE_MODEL_PATH)
        self.last_data_refresh = datetime.now()
        
        # User state storage
        self.user_data = {}
        
        # Start data refresh thread
        self.start_data_refresh_thread()
        
    def get_user_data(self, user_id):
        """Initialize or get user data"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'district': None,
                'cluster': None,
                'dislike_count': 0,
                'favorites': [],
                'shown_listings': [],
                'price_estimation': {},
                'current_menu': None
            }
        return self.user_data[user_id]
        
    def start_data_refresh_thread(self):
        """Start a thread to refresh data every 24 hours"""
        def refresh_data_task():
            while True:
                # Sleep for 24 hours (86400 seconds)
                time.sleep(86400)
                try:
                    # Reload the data
                    self.df = pd.read_csv(DATA_PATH)
                    self.last_data_refresh = datetime.now()
                    logging.info(f"Data refreshed successfully at {self.last_data_refresh}")
                except Exception as e:
                    logging.error(f"Error refreshing data: {e}")
        
        # Start the thread
        refresh_thread = threading.Thread(target=refresh_data_task, daemon=True)
        refresh_thread.start()
        logging.info("Data refresh thread started")
    
    def get_image_path(self, listing_id):
        """Find the image path for a given listing ID"""
        # Convert listing_id to string if it's not already
        listing_id_str = str(listing_id)
        
        # Look for a folder matching the listing ID pattern
        pattern = f"listing_{listing_id_str}"
        matching_folders = []
        
        for folder in os.listdir(IMAGES_DIR):
            if re.match(pattern, folder):
                matching_folders.append(folder)
        
        if matching_folders:
            folder_path = os.path.join(IMAGES_DIR, matching_folders[0])
            if os.path.isdir(folder_path):
                # Get the first image from the folder
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    return os.path.join(folder_path, image_files[0])
        
        return None
    
    def get_district_listings(self, district_code):
        """Get listings for a specific district"""
        return self.df[self.df['full_address_code'] == district_code]
    
    def get_cluster_listings(self, district_code, cluster):
        """Get listings for a specific district and cluster"""
        return self.df[(self.df['full_address_code'] == district_code) & (self.df['cluster'] == cluster)]
    
    def get_random_listing(self, user_id, district_code=None, cluster=None):
        """Get a random listing from specified district and cluster that hasn't been shown yet"""
        user_data = self.get_user_data(user_id)
        
        if district_code is None:
            district_code = user_data['district']
        
        # Filter by district and cluster if specified
        if cluster is not None:
            potential_listings = self.get_cluster_listings(district_code, cluster)
        else:
            potential_listings = self.get_district_listings(district_code)
        
        # Remove already shown listings
        potential_listings = potential_listings[~potential_listings['id'].isin(user_data['shown_listings'])]
        
        # If no more listings in cluster, reset to all listings in the district
        if potential_listings.empty and cluster is not None:
            potential_listings = self.get_district_listings(district_code)
            potential_listings = potential_listings[~potential_listings['id'].isin(user_data['shown_listings'])]
        
        # If still no listings, reset shown listings
        if potential_listings.empty:
            user_data['shown_listings'] = []
            if cluster is not None:
                potential_listings = self.get_cluster_listings(district_code, cluster)
            else:
                potential_listings = self.get_district_listings(district_code)
        
        # Get a random listing
        if not potential_listings.empty:
            listing = potential_listings.sample(1).iloc[0]
            user_data['shown_listings'].append(listing['id'])
            return listing
        
        return None
    
    def estimate_price(self, features):
        """Estimate rental price based on features"""
        # Create the features array for prediction
        X = np.array([
            features['floor'],
            features['total_floors'],
            features['area_sqm'],
            features['rooms'],
            features['bathroom_code']
        ]).reshape(1, -5)
        
        # Predict price
        predicted_price = self.price_model.predict(X)[0]
        
        # Create a price range (±15%)
        lower_bound = int(predicted_price * 0.85)
        upper_bound = int(predicted_price * 1.15)
        
        return int(predicted_price), lower_bound, upper_bound

# Bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user = update.effective_user
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    user_data['current_menu'] = 'main'
    
    # Create main menu keyboard
    keyboard = [
        [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
        [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        f"Привет, {user.first_name}! Я бот для поиска аренды квартир в Алматы.",
        reply_markup=reply_markup
    )
    
    return MAIN_MENU

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    # Create help message
    help_text = (
        "🤖 *Помощь по использованию бота*\n\n"
        "*Основные команды:*\n"
        "🏠 *Поиск квартиры* - начать поиск квартиры по районам\n"
        "❤️ *Избранное* - просмотр сохраненных объявлений\n"
        "💰 *Оценка стоимости* - рассчитать примерную стоимость аренды\n"
        "ℹ️ *Помощь* - показать эту справку\n\n"
        
        "*Как пользоваться поиском:*\n"
        "1. Выберите район Алматы\n"
        "2. Просматривайте объявления\n"
        "3. Нажмите 👍 если объявление нравится (бот покажет похожие)\n"
        "4. Нажмите 👎 если не нравится (бот покажет другие)\n"
        "5. Нажмите ❤️ чтобы добавить в избранное\n"
        "6. Используйте кнопку «Назад» для возврата в меню\n\n"
        
        "*Данные обновляются автоматически каждые 24 часа*\n"
        f"Последнее обновление: {bot.last_data_refresh.strftime('%d.%m.%Y %H:%M')}"
    )
    
    # Create main menu keyboard
    keyboard = [
        [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
        [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=reply_markup)
    
    return MAIN_MENU

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle main menu selections"""
    text = update.message.text
    
    if text == "🏠 Поиск квартиры":
        return await search_command(update, context)
    elif text == "❤️ Избранное":
        return await favorites_command(update, context)
    elif text == "💰 Оценка стоимости":
        return await estimate_command(update, context)
    elif text == "ℹ️ Помощь":
        return await help_command(update, context)
    else:
        # Return to main menu if unknown command
        return await start(update, context)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for search command"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    user_data['current_menu'] = 'search'
    
    # Create keyboard with district options
    keyboard = []
    row = []
    
    # Create rows with 2 buttons each
    for i, (district_name, district_code) in enumerate(DISTRICT_CHOICES.items()):
        row.append(InlineKeyboardButton(district_name, callback_data=f"district_{district_code}"))
        if len(row) == 2 or i == len(DISTRICT_CHOICES) - 1:
            keyboard.append(row)
            row = []
    
    # Add back button
    keyboard.append([InlineKeyboardButton("« Назад в меню", callback_data="back_to_menu")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Выберите район Алматы:",
        reply_markup=reply_markup
    )
    
    return DISTRICT_SELECTION

async def back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle back to menu button"""
    query = update.callback_query
    await query.answer()
    
    # Create main menu keyboard
    keyboard = [
        [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
        [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await query.message.reply_text(
        "Вы вернулись в главное меню. Выберите действие:",
        reply_markup=reply_markup
    )
    
    return MAIN_MENU

async def district_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle district selection"""
    query = update.callback_query
    await query.answer()
    
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    # Check if this is a "back to menu" action
    if query.data == "back_to_menu":
        return await back_to_menu(update, context)
        
    # Extract district code from callback data
    district_code = int(query.data.split('_')[1])
    user_data['district'] = district_code
    user_data['cluster'] = None
    user_data['dislike_count'] = 0
    user_data['shown_listings'] = []
    
    # Get a random listing from the selected district
    await show_listing(update, context)
    
    return VIEWING_LISTINGS

async def show_listing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show a random listing to the user"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    # Get listing based on current user state
    listing = bot.get_random_listing(
        user_id,
        district_code=user_data['district'],
        cluster=user_data['cluster']
    )
    
    if listing is None:
        if update.callback_query:
            await update.callback_query.message.reply_text(
                "Нет доступных объявлений. Попробуйте другой район."
            )
        else:
            await update.message.reply_text(
                "Нет доступных объявлений. Попробуйте другой район."
            )
        return
    
    # Get image for the listing
    image_path = bot.get_image_path(listing['id'])
    
    # Create listing details message
    details = (
        f"🏠 {listing['title']}\n\n"
        f"💰 Цена: {listing['price']:,} тг/месяц\n"
        f"🛏️ Комнат: {int(listing['rooms'])}\n"
        f"📏 Площадь: {listing['area_sqm']} м²\n"
        f"🏢 Этаж: {int(listing['floor'])}/{int(listing['total_floors'])}\n"
        f"🚿 Санузел: {BATHROOM_MAPPING.get(listing['bathroom_code'], 'неизвестно')}\n"
        f"🔗 Ссылка: {listing['url']}"
    )
    
    # Create keyboard
    keyboard = [
        [
            InlineKeyboardButton("👍 Нравится", callback_data=f"like_{listing['id']}"),
            InlineKeyboardButton("👎 Не нравится", callback_data="dislike"),
        ],
        [InlineKeyboardButton("❤️ Добавить в избранное", callback_data=f"favorite_{listing['id']}")],
        [InlineKeyboardButton("« Назад в меню", callback_data="back_to_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send message with image if available
    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as photo:
            if update.callback_query:
                await update.callback_query.message.reply_photo(
                    photo=photo,
                    caption=details,
                    reply_markup=reply_markup
                )
            else:
                await update.message.reply_photo(
                    photo=photo,
                    caption=details,
                    reply_markup=reply_markup
                )
    else:
        # Send without image if not available
        if update.callback_query:
            await update.callback_query.message.reply_text(
                details + "\n\n(Изображение недоступно)",
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                details + "\n\n(Изображение недоступно)",
                reply_markup=reply_markup
            )

async def listing_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle listing interactions (like, dislike, favorite)"""
    query = update.callback_query
    await query.answer()
    
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    # Check if this is a "back to menu" action
    if query.data == "back_to_menu":
        return await back_to_menu(update, context)
    
    action = query.data.split('_')[0]
    
    if action == "like":
        # User liked the listing, remember the cluster
        listing_id = int(query.data.split('_')[1])
        listing = bot.df[bot.df['id'] == listing_id].iloc[0]
        user_data['cluster'] = listing['cluster']
        user_data['dislike_count'] = 0
        await query.message.reply_text(f"👍 Вам понравилось это объявление! Показываю похожие варианты.")
    
    elif action == "dislike":
        # User disliked the listing
        user_data['dislike_count'] += 1
        
        # Reset cluster after 10 consecutive dislikes
        if user_data['dislike_count'] >= 10:
            user_data['cluster'] = None
            user_data['dislike_count'] = 0
            await query.message.reply_text("Похоже, вам не нравятся предложенные варианты. Начинаем поиск заново.")
    
    elif action == "favorite":
        # Add to favorites
        listing_id = int(query.data.split('_')[1])
        if listing_id not in user_data['favorites']:
            user_data['favorites'].append(listing_id)
            await query.message.reply_text("❤️ Объявление добавлено в избранное!")
    
    # Show next listing
    await show_listing(update, context)
    
    return VIEWING_LISTINGS

async def favorites_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user's favorite listings"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    user_data['current_menu'] = 'favorites'
    
    if not user_data['favorites']:
        # Create main menu keyboard
        keyboard = [
            [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
            [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            "У вас пока нет избранных объявлений.",
            reply_markup=reply_markup
        )
        return MAIN_MENU
    
    await update.message.reply_text(f"Ваши избранные объявления ({len(user_data['favorites'])}):")
    
    # Show each favorite listing
    for listing_id in user_data['favorites']:
        listing = bot.df[bot.df['id'] == listing_id]
        
        if listing.empty:
            continue
            
        listing = listing.iloc[0]
        
        # Get image for the listing
        image_path = bot.get_image_path(listing_id)
        
        # Create listing details message
        details = (
            f"🏠 {listing['title']}\n\n"
            f"💰 Цена: {listing['price']:,} тг/месяц\n"
            f"🛏️ Комнат: {int(listing['rooms'])}\n"
            f"📏 Площадь: {listing['area_sqm']} м²\n"
            f"🏢 Этаж: {int(listing['floor'])}/{int(listing['total_floors'])}\n"
            f"🚿 Санузел: {BATHROOM_MAPPING.get(listing['bathroom_code'], 'неизвестно')}\n"
            f"🔗 Ссылка: {listing['url']}"
        )
        
        # Create keyboard
        keyboard = [
            [InlineKeyboardButton("❌ Удалить из избранного", callback_data=f"remove_favorite_{listing_id}")],
            [InlineKeyboardButton("« Назад в меню", callback_data="back_to_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send message with image if available
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=details,
                    reply_markup=reply_markup
                )
        else:
            # Send without image if not available
            await update.message.reply_text(
                details + "\n\n(Изображение недоступно)",
                reply_markup=reply_markup
            )

async def remove_favorite_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Remove a listing from favorites"""
    query = update.callback_query
    await query.answer()
    
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    # Check if this is a "back to menu" action
    if query.data == "back_to_menu":
        return await back_to_menu(update, context)
    
    listing_id = int(query.data.split('_')[2])
    
    if listing_id in user_data['favorites']:
        user_data['favorites'].remove(listing_id)
        await query.message.reply_text("Объявление удалено из избранного.")
        # Delete the message with the removed favorite
        await query.message.delete()

# Price estimation handlers
async def estimate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start price estimation process"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    user_data['current_menu'] = 'estimate'
    
    # Reset price estimation data
    user_data['price_estimation'] = {}
    
    # Create cancel button
    keyboard = [[KeyboardButton("Отмена")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "Давайте оценим стоимость аренды квартиры. Ответьте на несколько вопросов.\n\n"
        "Введите этаж:",
        reply_markup=reply_markup
    )
    
    return PRICE_ESTIMATION_FLOOR

async def price_estimation_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle floor input for price estimation"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    try:
        floor = int(update.message.text)
        if floor <= 0:
            await update.message.reply_text("Этаж должен быть положительным числом. Пожалуйста, введите еще раз:")
            return PRICE_ESTIMATION_FLOOR
            
        user_data['price_estimation']['floor'] = floor
        
        await update.message.reply_text("Введите общее количество этажей в доме:")
        return PRICE_ESTIMATION_TOTAL_FLOORS
        
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return PRICE_ESTIMATION_FLOOR

async def price_estimation_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle total floors input for price estimation"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    try:
        total_floors = int(update.message.text)
        floor = user_data['price_estimation']['floor']
        
        if total_floors <= 0:
            await update.message.reply_text("Общее количество этажей должно быть положительным числом. Пожалуйста, введите еще раз:")
            return PRICE_ESTIMATION_TOTAL_FLOORS
            
        if floor > total_floors:
            await update.message.reply_text(f"Этаж ({floor}) не может быть больше общего количества этажей. Пожалуйста, введите корректное значение:")
            return PRICE_ESTIMATION_TOTAL_FLOORS
            
        user_data['price_estimation']['total_floors'] = total_floors
        
        await update.message.reply_text("Введите площадь квартиры (в квадратных метрах):")
        return PRICE_ESTIMATION_AREA
        
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return PRICE_ESTIMATION_TOTAL_FLOORS

async def price_estimation_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle area input for price estimation"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    try:
        area = float(update.message.text)
        if area <= 0:
            await update.message.reply_text("Площадь должна быть положительным числом. Пожалуйста, введите еще раз:")
            return PRICE_ESTIMATION_AREA
            
        user_data['price_estimation']['area_sqm'] = area
        
        await update.message.reply_text("Введите количество комнат:")
        return PRICE_ESTIMATION_ROOMS
        
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return PRICE_ESTIMATION_AREA

async def price_estimation_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle rooms input for price estimation"""
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    try:
        rooms = int(update.message.text)
        if rooms <= 0:
            await update.message.reply_text("Количество комнат должно быть положительным числом. Пожалуйста, введите еще раз:")
            return PRICE_ESTIMATION_ROOMS
            
        user_data['price_estimation']['rooms'] = rooms
        
        # Create keyboard for bathroom selection
        keyboard = [
            [InlineKeyboardButton("2 с/у и более", callback_data="bathroom_0")],
            [InlineKeyboardButton("Раздельный", callback_data="bathroom_2")],
            [InlineKeyboardButton("Совмещенный", callback_data="bathroom_5")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Выберите тип санузла:",
            reply_markup=reply_markup
        )
        return PRICE_ESTIMATION_BATHROOM
        
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите число. Попробуйте еще раз:")
        return PRICE_ESTIMATION_ROOMS

async def price_estimation_bathroom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle bathroom selection for price estimation"""
    query = update.callback_query
    await query.answer()
    
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    bathroom_code = int(query.data.split('_')[1])
    user_data['price_estimation']['bathroom_code'] = bathroom_code
    
    # Now we have all the features, estimate the price
    predicted_price, lower_bound, upper_bound = bot.estimate_price(user_data['price_estimation'])
    
    # Format the results
    result_text = (
        "📊 *Результаты оценки стоимости аренды*\n\n"
        f"🏢 Этаж: {user_data['price_estimation']['floor']}/{user_data['price_estimation']['total_floors']}\n"
        f"📏 Площадь: {user_data['price_estimation']['area_sqm']} м²\n"
        f"🛏️ Комнат: {user_data['price_estimation']['rooms']}\n"
        f"🚿 Санузел: {BATHROOM_MAPPING[bathroom_code]}\n\n"
        f"💰 *Предполагаемая стоимость аренды:*\n"
        f"➡️ {predicted_price:,} тг/месяц\n\n"
        f"💸 *Диапазон цен:*\n"
        f"➡️ {lower_bound:,} - {upper_bound:,} тг/месяц"
    )
    
    # Create main menu keyboard
    keyboard = [
        [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
        [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await query.message.reply_text(
        result_text,
        parse_mode='Markdown',
        reply_markup=reply_markup
    )
    
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):

    """canceling logic"""
  
    
    # Create main menu keyboard
    keyboard = [
        [KeyboardButton("🏠 Поиск квартиры"), KeyboardButton("❤️ Избранное")],
        [KeyboardButton("💰 Оценка стоимости"), KeyboardButton("ℹ️ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "Операция отменена. Возвращаемся в главное меню.",
        reply_markup=reply_markup
    )
    return MAIN_MENU

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages"""
    text = update.message.text
    
    # Handle "Cancel" message
    if text == "Отмена":
        return await cancel(update, context)
    
    # Handle main menu commands
    bot = context.bot_data.get('krisha_bot')
    user_id = update.effective_user.id
    user_data = bot.get_user_data(user_id)
    
    if user_data.get('current_menu') == 'main':
        return await main_menu_handler(update, context)
    else:
        # Return to main menu for any other text
        return await start(update, context)



def main():
    """Run the bot"""
    # Create the bot instance
    krisha_bot = KrishaBot()
    
    # Create application
    application = Application.builder().token("your_token").build()
    
    # Store the bot instance
    application.bot_data['krisha_bot'] = krisha_bot
    
    # Main conversation handler
    main_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu_handler)
            ],
            DISTRICT_SELECTION: [
                CallbackQueryHandler(district_callback, pattern=r"^district_"),
                CallbackQueryHandler(back_to_menu, pattern=r"^back_to_menu$")
            ],
            VIEWING_LISTINGS: [
                CallbackQueryHandler(listing_callback, pattern=r"^like_|^dislike$|^favorite_"),
                CallbackQueryHandler(back_to_menu, pattern=r"^back_to_menu$")
            ],
            PRICE_ESTIMATION_FLOOR: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, price_estimation_floor)
            ],
            PRICE_ESTIMATION_TOTAL_FLOORS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, price_estimation_total_floors)
            ],
            PRICE_ESTIMATION_AREA: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, price_estimation_area)
            ],
            PRICE_ESTIMATION_ROOMS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, price_estimation_rooms)
            ],
            PRICE_ESTIMATION_BATHROOM: [
                CallbackQueryHandler(price_estimation_bathroom, pattern=r"^bathroom_")
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    # Add handlers
    application.add_handler(main_conv_handler)
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("favorites", favorites_command))
    application.add_handler(CommandHandler("estimate", estimate_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(remove_favorite_callback, pattern=r"^remove_favorite_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    # Log startup
    logging.info("Starting bot...")
    
    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()
