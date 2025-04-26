import os
import pandas as pd
import numpy as np
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
import logging
from datetime import datetime, timedelta
import random
import asyncio

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# States for the conversation
DISTRICT_SELECT, VIEWING_LISTINGS = range(2)
# States for price estimation
FLOOR_INPUT, TOTAL_FLOORS_INPUT, AREA_INPUT, ROOMS_INPUT, BATHROOM_INPUT = range(2, 7)

# Path constants
DATA_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\krisha_bot\data\regular_scrapping\cleaned\almaty_apartments_cleaned.csv"
MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\krisha_bot\models\krisha_almaty_rental_kmeans29_pipeline.joblib"
PRICE_MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\krisha_bot\models\krisha_almaty_rental_stacking.joblib"

# Bot token - place your token directly here or use environment variables
TOKEN = "your_token"
# Alternatively, use environment variables (more secure):
# TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# User preferences storage
user_preferences = {}  # Will store user_id -> {district, preferred_cluster}

# Mapping dictionaries for categorical values (reverse mappings)
BATHROOM_MAPPING = {0: '2 с/у и более', 1: 'неизвестно', 2: 'разделен', 3: 'разделен, совмещен', 
                   4: 'раздельный', 5: 'совмещен', 6: 'совмещенный'}

# Bathroom encoding for price estimation as specified
BATHROOM_ENCODING = {
    '2 с/у и более': 0,
    'разделен': 2,
    'совмещен': 5
}

PARKING_MAPPING = {0: 'гараж', 1: 'неизвестно', 2: 'паркинг', 3: 'рядом охраняемая стоянка'}

# Full address mapping with the correct newline character
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

# For district selection in UI - simple district names for buttons
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

# Security feature mapping simplification
def format_security(security_code):
    security_features = []
    
    # Extract the key features based on common elements in security codes
    if 'охрана' in str(security_code):
        security_features.append('Охрана')
    if 'домофон' in str(security_code):
        security_features.append('Домофон')
    if 'видеонаблюдение' in str(security_code):
        security_features.append('Видеонаблюдение')
    if 'видеодомофон' in str(security_code):
        security_features.append('Видеодомофон')
    if 'консьерж' in str(security_code):
        security_features.append('Консьерж')
    if 'решетки на окнах' in str(security_code):
        security_features.append('Решетки на окнах')
    if 'сигнализация' in str(security_code):
        security_features.append('Сигнализация')
    if 'кодовый замок' in str(security_code):
        security_features.append('Кодовый замок')
    
    if not security_features:
        return "Неизвестно"
    
    return ", ".join(security_features)

# Simplified furniture descriptions
def format_furniture(furniture_code):
    furniture = set()
    
    if furniture_code == 34 or furniture_code == 1:
        return "Неизвестно"
    
    # Common furniture items to check for
    if 'кровать' in str(furniture_code):
        furniture.add('Кровать')
    if 'диван' in str(furniture_code):
        furniture.add('Диван')
    if 'шкаф' in str(furniture_code):
        furniture.add('Шкаф')
    if 'кухонный гарнитур' in str(furniture_code):
        furniture.add('Кухонный гарнитур')
    if 'обеденный стол' in str(furniture_code):
        furniture.add('Обеденный стол')
    if 'рабочий стол' in str(furniture_code):
        furniture.add('Рабочий стол')
    
    if not furniture:
        return "Неизвестно"
    
    return ", ".join(furniture)

# Class to store and manage the dataset
class DataManager:
    def __init__(self):
        self.data = None
        self.model = None
        self.price_model = None
        self.last_loaded = None
        self.load_data_and_model()
    
    def load_data_and_model(self):
        """Load the dataset and machine learning models"""
        try:
            # Check if paths exist
            if not os.path.exists(DATA_PATH):
                logger.error(f"Data file not found at: {DATA_PATH}")
                return False
                
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found at: {MODEL_PATH}")
                return False
                
            if not os.path.exists(PRICE_MODEL_PATH):
                logger.error(f"Price model file not found at: {PRICE_MODEL_PATH}")
                return False
                
            self.data = pd.read_csv(DATA_PATH)
            
            # Add unique IDs to listings if they don't have them
            if 'id' not in self.data.columns:
                self.data['id'] = range(1, len(self.data) + 1)
                
            self.model = joblib.load(MODEL_PATH)
            self.price_model = joblib.load(PRICE_MODEL_PATH)
            self.last_loaded = datetime.now()
            
            # Assign clusters to listings
            input_features = self.data[['floor', 'total_floors', 'area_sqm', 'rooms', 'price',
                                      'full_address_code', 'furniture_code', 'parking_code', 
                                      'security_code', 'bathroom_code']]
            self.data['cluster'] = self.model.predict(input_features)
            
            logger.info(f"Data loaded with {len(self.data)} listings and clusters assigned")
            return True
        except Exception as e:
            logger.error(f"Error loading data or model: {e}")
            return False
    
    def check_and_reload(self):
        """Check if data needs to be reloaded (every 24 hours)"""
        now = datetime.now()
        
        # Also reload at 7:00 AM as currently implemented
        is_reload_time = now.hour == 7 and 0 <= now.minute < 5
        
        # Add a time-based check (reload if data is older than 24 hours)
        time_to_reload = False
        if self.last_loaded:
            time_since_reload = now - self.last_loaded
            time_to_reload = time_since_reload.total_seconds() > 86400  # 24 hours in seconds
        
        if is_reload_time or time_to_reload:
            logger.info("Reloading data and model...")
            success = self.load_data_and_model()
            if success:
                self.last_loaded = now
                logger.info("Data and model successfully reloaded")
            else:
                logger.error("Failed to reload data and model")

    
    def get_random_listing_from_district(self, district_code, preferred_cluster=None):
        """Get a random listing from the specified district and optionally from preferred cluster"""
        self.check_and_reload()
        
        if self.data is None:
            logger.error("No data available for listings")
            return None
        
        district_listings = self.data[self.data['full_address_code'] == district_code].copy()
        
        if len(district_listings) == 0:
            return None
        
        # If preferred cluster is provided and listings exist in that cluster
        if preferred_cluster is not None:
            cluster_listings = district_listings[district_listings['cluster'] == preferred_cluster]
            if len(cluster_listings) > 0:
                logger.info(f"Found {len(cluster_listings)} listings in preferred cluster {preferred_cluster}")
                return cluster_listings.sample(1).iloc[0]
            else:
                logger.info(f"No listings in preferred cluster {preferred_cluster}, using random district listing instead")
        
        # Return random listing from district
        return district_listings.sample(1).iloc[0]
    
    def get_district_name(self, district_code):
        """Get district name from code"""
        return DISTRICT_MAPPING.get(district_code, "Неизвестный район")
    
    def estimate_price(self, floor, total_floors, area_sqm, rooms, bathroom_code):
        """Estimate price using the stacking model"""
        if self.price_model is None:
            logger.error("Price model not loaded")
            return None
        
        try:
            # Create input features in correct order
            input_features = np.array([[floor, total_floors, area_sqm, rooms, bathroom_code]])
            
            # Predict price
            estimated_price = self.price_model.predict(input_features)[0]
            
            # Convert to reasonable integer
            return int(estimated_price)
        except Exception as e:
            logger.error(f"Error estimating price: {e}")
            return None

# Initialize data manager
data_manager = DataManager()

# Function to handle stop button action
async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the stop button press"""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        "✅ Операция прервана. Вы можете использовать команды:\n"
        "/seerent - для поиска квартир\n"
        "/estimate - для оценки стоимости\n"
        "/help - для получения справки"
    )
    
    return ConversationHandler.END

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the conversation and provide bot description"""
    user = update.effective_user
    
    # Initialize user data
    context.user_data.clear()
    
    await update.message.reply_text(
        f"👋 Привет, {user.first_name}! Добро пожаловать в @krisha_home_bot! 🏠\n\n"
        f"Я ваш личный помощник в поиске квартиры для аренды в Алматы. Вот что я умею:\n\n"
        f"🔍 Показываю актуальные предложения по аренде квартир\n"
        f"🏙️ Фильтрую объявления по районам города\n"
        f"🧠 Запоминаю ваши предпочтения и показываю похожие варианты\n"
        f"🔗 Предоставляю ссылки на понравившиеся объявления\n"
        f"💰 Оцениваю ориентировочную стоимость аренды вашей квартиры, 5 простых вопросов!\n"
        f"🔔 Отправляю уведомления о новых интересных предложениях\n\n"
        f"Чтобы начать поиск квартиры, используйте команду /seerent\n"
        f"Для оценки стоимости аренды, используйте команду /estimate\n"
        f"Для просмотра всех доступных команд, введите /help"
    )
    
    return ConversationHandler.END

async def show_district_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show district selection buttons"""
    # Create district selection keyboard
    keyboard = []
    row = []
    for i, (district_name, district_code) in enumerate(DISTRICT_CHOICES.items()):
        row.append(InlineKeyboardButton(district_name, callback_data=f"district_{district_code}"))
        if (i + 1) % 2 == 0 or i == len(DISTRICT_CHOICES) - 1:
            keyboard.append(row)
            row = []
    
    # Add a Stop button
    keyboard.append([InlineKeyboardButton("❌ Отменить", callback_data="stop")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text("Выберите район:", reply_markup=reply_markup)
        else:
            await update.message.reply_text("Выберите район:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error showing district selection: {e}")
        # Fallback to sending a new message if edit fails
        if update.callback_query:
            await update.callback_query.message.reply_text("Выберите район:", reply_markup=reply_markup)
    
    return DISTRICT_SELECT

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued"""
    await update.message.reply_text(
        "📚 Доступные команды бота:\n\n"
        "🏠 /start - Познакомиться с ботом и узнать его возможности\n"
        "🔍 /seerent - Начать поиск квартир для аренды\n"
        "💰 /estimate - Оценить стоимость аренды квартиры за 5 простых вопросов\n"
        "🔔 /notifications - Управление уведомлениями о новых предложениях\n"
        "❓ /help - Показать эту справку\n\n"
        "\nБот запоминает ваши предпочтения и показывает похожие квартиры! 😉"
    )

async def seerent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the rental viewing process"""
    user = update.effective_user
    
    # Initialize user data
    context.user_data.clear()
    
    await update.message.reply_text(
        f"🔍 Отлично, {user.first_name}! Давайте найдем для вас идеальную квартиру в Алматы. "
        f"\nВыберите район, в котором хотите искать квартиру:"
    )
    
    return await show_district_selection(update, context)

async def select_district(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle district selection"""
    query = update.callback_query
    await query.answer()
    
    district_code = int(query.data.split('_')[1])
    context.user_data['district'] = district_code
    
    # Reset preferred cluster when changing districts
    if 'preferred_cluster' in context.user_data:
        del context.user_data['preferred_cluster']
    
    # Initialize dislike counter when selecting district
    context.user_data['dislike_counter'] = 0
    
    district_name = data_manager.get_district_name(district_code)
    
    await query.edit_message_text(f"Вы выбрали район: {district_name}")
    
    # Show first listing
    return await show_listing(update, context)

def format_listing_details(listing):
    """Format listing details for display"""
    district = DISTRICT_MAPPING.get(listing['full_address_code'], "Неизвестно")
    
    # Format price with spaces for better readability
    price_str = f"{int(listing['price']):,}".replace(',', ' ')
    
    # Build the listing message
    details = [
        f"<b>Район:</b> {district.split('Алматы, ')[-1] if 'Алматы, ' in district else district}",
        f"<b>Цена:</b> {price_str} тенге",
        f"<b>Комнат:</b> {int(listing['rooms'])}",
        f"<b>Площадь:</b> {listing['area_sqm']} м²",
        f"<b>Этаж:</b> {int(listing['floor'])}/{int(listing['total_floors'])}",
        f"<b>Санузел:</b> {BATHROOM_MAPPING.get(listing['bathroom_code'], 'Неизвестно')}",
        f"<b>Паркинг:</b> {PARKING_MAPPING.get(listing['parking_code'], 'Неизвестно')}",
    ]
    
    # Add title and contact if available
    if 'title' in listing and not pd.isna(listing['title']):
        details.append(f"<b>Название:</b> {listing['title']}")
    
    if 'contact_name' in listing and not pd.isna(listing['contact_name']):
        details.append(f"<b>Контакт:</b> {listing['contact_name']}")
    
    return "\n".join(details)

async def show_listing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show a listing to the user"""
    district_code = context.user_data.get('district')
    preferred_cluster = context.user_data.get('preferred_cluster')
    
    # Get random listing from this district (and cluster if available)
    listing = data_manager.get_random_listing_from_district(district_code, preferred_cluster)
    
    # Make sure we don't show the same listing twice in a row
    current_listing_id = context.user_data.get('current_listing', {}).get('id', None)
    attempts = 0
    max_attempts = 5  # Prevent infinite loop
    
    while listing is not None and 'id' in listing and listing['id'] == current_listing_id and attempts < max_attempts:
        listing = data_manager.get_random_listing_from_district(district_code, preferred_cluster)
        attempts += 1
    
    if listing is None:
        message_text = "😕 К сожалению, не найдено объявлений в этом районе. Выберите другой район:"
        try:
            if update.callback_query:
                await update.callback_query.edit_message_text(message_text)
            else:
                await update.effective_message.reply_text(message_text)
        except Exception as e:
            logger.error(f"Error showing no listings message: {e}")
            if update.callback_query:
                await update.callback_query.message.reply_text(message_text)
            
        return await show_district_selection(update, context)
    
    # Store the current listing in context for later reference
    context.user_data['current_listing'] = listing.to_dict()
    
    # Format the listing details
    listing_text = format_listing_details(listing)
    
    # Add timestamp to make sure the message content is unique
    unique_indicator = f"\n\n<i>Показано: {datetime.now().strftime('%H:%M:%S')}</i>"
    listing_text += unique_indicator
    
    # Create keyboard with like/dislike buttons and stop button
    keyboard = [
        [
            InlineKeyboardButton("👎 Не нравится", callback_data="dislike"),
            InlineKeyboardButton("👍 Нравится, показать ссылку", callback_data="like")
        ],
        [InlineKeyboardButton("🔄 Изменить район", callback_data="change_district")],
        [InlineKeyboardButton("❌ Отменить поиск", callback_data="stop")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=listing_text,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
        else:
            await update.effective_message.reply_text(
                text=listing_text,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
    except Exception as e:
        logger.error(f"Error showing listing: {e}")
        # If editing fails, send a new message instead
        if update.callback_query:
            await update.callback_query.message.reply_text(
                text=f"Новое предложение:\n\n{listing_text}",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
    
    return VIEWING_LISTINGS

# New function to schedule notifications
async def schedule_listing_notification(context: ContextTypes.DEFAULT_TYPE, user_id, district_code, cluster):
    """Schedule a notification with a listing from the preferred cluster"""
    # Calculate random time within the next 24 hours
    # Random between 3 and 24 hours to avoid immediate notifications
    hours_delay = random.randint(3, 24)
    minutes_delay = random.randint(0, 59)
    seconds_delay = random.randint(0, 59)
    
    notification_time = datetime.now() + timedelta(
        hours=hours_delay, 
        minutes=minutes_delay,
        seconds=seconds_delay
    )
    
    # Store the job in user preferences
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    
    user_preferences[user_id]['district'] = district_code
    user_preferences[user_id]['preferred_cluster'] = cluster
    user_preferences[user_id]['notification_scheduled'] = True
    
    logger.info(
        f"Scheduled notification for user {user_id} at {notification_time} "
        f"(district: {district_code}, cluster: {cluster})"
    )
    
    # Schedule the job to run at the calculated time
    context.job_queue.run_once(
        send_listing_notification,
        notification_time - datetime.now(),
        data={'user_id': user_id, 'district_code': district_code, 'cluster': cluster},
        name=f"notification_{user_id}"
    )

# Function to send a notification with a listing
async def send_listing_notification(context: ContextTypes.DEFAULT_TYPE):
    """Send a notification with a listing from the preferred cluster"""
    job_data = context.job.data
    user_id = job_data['user_id']
    district_code = job_data['district_code']
    cluster = job_data['cluster']
    
    # Get a random listing from the preferred cluster
    listing = data_manager.get_random_listing_from_district(district_code, cluster)
    
    if listing is not None:
        # Format the listing details
        listing_text = format_listing_details(listing)
        district_name = data_manager.get_district_name(district_code)
        
        # Add notification header
        message = (
            f"🔔 <b>Новое интересное предложение!</b>\n\n"
            f"Я нашел для вас подходящую квартиру в районе {district_name.split('Алматы, ')[-1] if 'Алматы, ' in district_name else district_name}:\n\n"
            f"{listing_text}"
        )
        
        # Create keyboard with the link and continue buttons
        keyboard = [
            [InlineKeyboardButton("🔗 Посмотреть объявление", url=listing['url'])],
            [InlineKeyboardButton("🔍 Продолжить поиск", callback_data="continue_search")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            # Send the notification
            await context.bot.send_message(
                chat_id=user_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
            logger.info(f"Sent notification with listing to user {user_id}")
            
            # Schedule the next notification (recursively continue the notifications)
            # Schedule it with a different delay for variety
            await schedule_listing_notification(context, user_id, district_code, cluster)
            
        except Exception as e:
            logger.error(f"Error sending notification to user {user_id}: {e}")
            # If the notification fails (e.g., user blocked the bot), remove from preferences
            if user_id in user_preferences:
                user_preferences[user_id]['notification_scheduled'] = False
    else:
        logger.error(f"No listings found for notification to user {user_id} (district: {district_code}, cluster: {cluster})")

# Handler for the continue_search button from notifications
async def handle_continue_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle continue search button from notifications"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    # Check if user has preferences stored
    if user_id in user_preferences:
        district_code = user_preferences[user_id].get('district')
        preferred_cluster = user_preferences[user_id].get('preferred_cluster')
        
        # Store in context for the current session
        context.user_data['district'] = district_code
        context.user_data['preferred_cluster'] = preferred_cluster
        context.user_data['dislike_counter'] = 0
        
        # Continue to show listings
        await query.edit_message_text("Продолжаем поиск с вашими предпочтениями...")
        
        # Show a listing based on preferences
        listing = data_manager.get_random_listing_from_district(district_code, preferred_cluster)
        
        if listing is not None:
            # Store the current listing
            context.user_data['current_listing'] = listing.to_dict()
            
            # Format and show the listing
            listing_text = format_listing_details(listing)
            
            # Add timestamp
            unique_indicator = f"\n\n<i>Показано: {datetime.now().strftime('%H:%M:%S')}</i>"
            listing_text += unique_indicator
            
            # Create keyboard
            keyboard = [
                [
                    InlineKeyboardButton("👎 Не нравится", callback_data="dislike"),
                    InlineKeyboardButton("👍 Нравится, показать ссылку", callback_data="like")
                ],
                [InlineKeyboardButton("🔄 Изменить район", callback_data="change_district")],
                [InlineKeyboardButton("❌ Отменить поиск", callback_data="stop")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=user_id,
                text=listing_text,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
            
            return VIEWING_LISTINGS
        else:
            # No listings found
            await query.edit_message_text(
                "😕 К сожалению, не найдено подходящих объявлений. "
                "Попробуйте изменить параметры поиска с помощью команды /seerent"
            )
            return ConversationHandler.END
    else:
        # No preferences stored
        await query.edit_message_text(
            "Для продолжения поиска начните с команды /seerent"
        )
        return ConversationHandler.END

async def handle_listing_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user's response to a listing"""
    query = update.callback_query
    await query.answer()
    
    user_response = query.data
    user_id = update.effective_user.id
    
    if user_response == "stop":
        # User wants to stop the process
        return await handle_stop(update, context)
    
    if user_response == "change_district":
        # User wants to change district
        # Reset preferred cluster when changing districts
        if 'preferred_cluster' in context.user_data:
            del context.user_data['preferred_cluster']
        # Reset dislike counter when changing districts
        context.user_data['dislike_counter'] = 0
        return await show_district_selection(update, context)
    
    elif user_response == "dislike":
        # User didn't like this listing
        # Increment dislike counter
        dislike_counter = context.user_data.get('dislike_counter', 0) + 1
        context.user_data['dislike_counter'] = dislike_counter
        
        # If user disliked 10 times in a row, reset preferred cluster
        if dislike_counter >= 10 and 'preferred_cluster' in context.user_data:
            del context.user_data['preferred_cluster']
            context.user_data['dislike_counter'] = 0  # Reset counter
            
            # Inform user that we're showing different recommendations now
            try:
                await query.edit_message_text(
                    "🔄 Понял, видимо эти варианты вам не подходят. Сейчас покажу другие предложения...",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("👍 Хорошо", callback_data="acknowledge_change")]
                    ])
                )
                await asyncio.sleep(2)  # Brief pause
            except Exception as e:
                logger.error(f"Error showing change notification: {e}")
    
    elif user_response == "like":
        # User liked this listing
        # Store their preference
        if 'current_listing' in context.user_data:
            current_listing = context.user_data['current_listing']
            if 'cluster' in current_listing:
                # Store the preferred cluster for this user
                preferred_cluster = current_listing['cluster']
                context.user_data['preferred_cluster'] = preferred_cluster
                
                # Store in global preferences dictionary for notifications
                if user_id not in user_preferences:
                    user_preferences[user_id] = {}
                
                user_preferences[user_id]['district'] = context.user_data.get('district')
                user_preferences[user_id]['preferred_cluster'] = preferred_cluster
                
                # Reset dislike counter when user likes a listing
                context.user_data['dislike_counter'] = 0
                
                # Schedule notification if not already scheduled
                if not user_preferences[user_id].get('notification_scheduled', False):
                    await schedule_listing_notification(
                        context, 
                        user_id, 
                        context.user_data.get('district'), 
                        preferred_cluster
                    )
            
            # Send link to the listing
            if 'url' in current_listing and current_listing['url']:
                keyboard = [
                    [InlineKeyboardButton("🔗 Перейти к объявлению", url=current_listing['url'])],
                    [InlineKeyboardButton("🔍 Продолжить поиск", callback_data="continue")],
                    [InlineKeyboardButton("❌ Завершить поиск", callback_data="stop")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    "👍 Отлично! Вот ссылка на это объявление:",
                    reply_markup=reply_markup
                )
                return VIEWING_LISTINGS
            else:
                await query.edit_message_text(
                    "😕 К сожалению, у этого объявления нет доступной ссылки. Хотите продолжить поиск?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔍 Продолжить поиск", callback_data="continue")],
                        [InlineKeyboardButton("❌ Завершить поиск", callback_data="stop")]
                    ])
                )
                return VIEWING_LISTINGS
    
    elif user_response == "continue" or user_response == "acknowledge_change":
        # User wants to continue viewing listings
        return await show_listing(update, context)
    
    # Default action - show next listing
    return await show_listing(update, context)

# Price estimation conversation handlers
async def start_price_estimation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the price estimation process"""
    # Reset estimation data
    context.user_data.clear()
    
    await update.message.reply_text(
        "💰 Давайте оценим стоимость аренды вашей квартиры!\n\n"
        "Для точной оценки мне нужно задать несколько вопросов.\n\n"
        "Для начала, укажите этаж квартиры:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Отменить", callback_data="stop")]
        ])
    )
    
    return FLOOR_INPUT

async def process_floor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process floor input and ask for total floors"""
    # Check if this is a callback query (from stop button)
    if update.callback_query:
        query = update.callback_query
        if query.data == "stop":
            return await handle_stop(update, context)
    
    try:
        floor = int(update.message.text.strip())
        if floor <= 0:
            await update.message.reply_text(
                "Этаж должен быть положительным числом. Пожалуйста, введите корректное значение:"
            )
            return FLOOR_INPUT
        
        context.user_data['floor'] = floor
        
        await update.message.reply_text(
            f"Отлично! Теперь укажите общее количество этажей в доме:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отменить", callback_data="stop")]
            ])
        )
        return TOTAL_FLOORS_INPUT
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное числовое значение для этажа:"
        )
        return FLOOR_INPUT

async def process_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process total floors input and ask for area"""
    # Check if this is a callback query (from stop button)
    if update.callback_query:
        query = update.callback_query
        if query.data == "stop":
            return await handle_stop(update, context)
    
    try:
        total_floors = int(update.message.text.strip())
        if total_floors <= 0:
            await update.message.reply_text(
                "Общее количество этажей должно быть положительным числом. Пожалуйста, введите корректное значение:"
            )
            return TOTAL_FLOORS_INPUT
        
        floor = context.user_data.get('floor', 0)
        if floor > total_floors:
            await update.message.reply_text(
                f"Этаж квартиры ({floor}) не может быть больше общего количества этажей в доме. "
                f"Пожалуйста, введите корректное значение:"
            )
            return TOTAL_FLOORS_INPUT
        
        context.user_data['total_floors'] = total_floors
        
        await update.message.reply_text(
            f"Теперь укажите общую площадь квартиры в квадратных метрах:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отменить", callback_data="stop")]
            ])
        )
        return AREA_INPUT
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное числовое значение для общего количества этажей:"
        )
        return TOTAL_FLOORS_INPUT

async def process_area(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process area input and ask for rooms"""
    # Check if this is a callback query (from stop button)
    if update.callback_query:
        query = update.callback_query
        if query.data == "stop":
            return await handle_stop(update, context)
    
    try:
        area = float(update.message.text.strip().replace(',', '.'))
        if area <= 0:
            await update.message.reply_text(
                "Площадь должна быть положительным числом. Пожалуйста, введите корректное значение:"
            )
            return AREA_INPUT
        
        context.user_data['area_sqm'] = area
        
        await update.message.reply_text(
            f"Отлично! Теперь укажите количество комнат в квартире:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Отменить", callback_data="stop")]
            ])
        )
        return ROOMS_INPUT
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное числовое значение для площади:"
        )
        return AREA_INPUT

async def process_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process rooms input and ask for bathroom type"""
    # Check if this is a callback query (from stop button)
    if update.callback_query:
        query = update.callback_query
        if query.data == "stop":
            return await handle_stop(update, context)
    
    try:
        rooms = int(update.message.text.strip())
        if rooms <= 0:
            await update.message.reply_text(
                "Количество комнат должно быть положительным числом. Пожалуйста, введите корректное значение:"
            )
            return ROOMS_INPUT
        
        context.user_data['rooms'] = rooms
        
        # Create keyboard for bathroom selection
        keyboard = []
        for bathroom_name, code in BATHROOM_ENCODING.items():
            keyboard.append([InlineKeyboardButton(bathroom_name, callback_data=f"bathroom_{code}")])
        
        # Add stop button
        keyboard.append([InlineKeyboardButton("❌ Отменить", callback_data="stop")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"И последний вопрос - выберите тип санузла:",
            reply_markup=reply_markup
        )
        return BATHROOM_INPUT
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное числовое значение для количества комнат:"
        )
        return ROOMS_INPUT

async def process_bathroom(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process bathroom selection and calculate price estimate"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "stop":
        return await handle_stop(update, context)
    
    bathroom_code = int(query.data.split('_')[1])
    context.user_data['bathroom_code'] = bathroom_code
    
    # Get all the inputs
    floor = context.user_data.get('floor')
    total_floors = context.user_data.get('total_floors')
    area_sqm = context.user_data.get('area_sqm')
    rooms = context.user_data.get('rooms')
    
    # Calculate price estimate
    estimated_price = data_manager.estimate_price(
        floor, total_floors, area_sqm, rooms, bathroom_code
    )


    if estimated_price is not None:

        # Format price with spaces for better readability
        formatted_price = f"{estimated_price:,}".replace(',', ' ')
        
        # Create a ±10% price range
        lower_price = int(estimated_price * 0.9)
        upper_price = int(estimated_price * 1.1)
        formatted_lower = f"{lower_price:,}".replace(',', ' ')
        formatted_upper = f"{upper_price:,}".replace(',', ' ')
        
        # Format price with spaces
        price_str = f"{estimated_price:,}".replace(',', ' ')
        
        # Create message with all details
        message = (
            f"💰 <b>Оценка стоимости аренды</b>\n\n"
            f"На основе предоставленных данных, ориентировочная стоимость аренды вашей квартиры составляет:\n\n"
            f"<b>{price_str} тенге в месяц</b>\n\n"
            f"<b>Возможный диапазон:</b> {formatted_lower} - {formatted_upper} тенге/месяц\n"
            f"<b>Параметры квартиры:</b>\n"
            f"- Этаж: {floor}/{total_floors}\n"
            f"- Площадь: {area_sqm} м²\n"
            f"- Комнат: {rooms}\n"
            f"- Санузел: {list(BATHROOM_ENCODING.keys())[list(BATHROOM_ENCODING.values()).index(bathroom_code)]}\n\n"
            f"Хотите посмотреть актуальные предложения по аренде квартир?"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔍 Посмотреть предложения", callback_data="start_search")],
            [InlineKeyboardButton("🏠 Вернуться в главное меню", callback_data="stop")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        
        return VIEWING_LISTINGS
    else:
        await query.edit_message_text(
            "😕 К сожалению, не удалось оценить стоимость с указанными параметрами. "
            "Попробуйте еще раз с другими параметрами."
        )
        return ConversationHandler.END

async def handle_start_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle transition from price estimation to listing search"""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text("Отлично! Давайте найдем для вас подходящую квартиру.")
    
    return await show_district_selection(update, context)

async def notification_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle notification settings"""
    user_id = update.effective_user.id
    
    if user_id in user_preferences and user_preferences[user_id].get('notification_scheduled', False):
        # User has notifications enabled
        keyboard = [
            [InlineKeyboardButton("❌ Отключить уведомления", callback_data="disable_notifications")],
            [InlineKeyboardButton("🔄 Изменить предпочтения", callback_data="change_preferences")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        district_code = user_preferences[user_id].get('district')
        district_name = data_manager.get_district_name(district_code)
        
        await update.message.reply_text(
            f"🔔 У вас включены уведомления о новых квартирах в районе {district_name.split('Алматы, ')[-1] if 'Алматы, ' in district_name else district_name}.\n\n"
            f"Я буду периодически отправлять вам интересные варианты, соответствующие вашим предпочтениям.",
            reply_markup=reply_markup
        )
    else:
        # User does not have notifications enabled
        keyboard = [
            [InlineKeyboardButton("🔔 Включить уведомления", callback_data="enable_notifications")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📢 В данный момент уведомления о новых квартирах отключены.\n\n"
            "Чтобы включить уведомления, сначала найдите понравившуюся квартиру через /seerent, "
            "а затем нажмите '👍 Нравится'.",
            reply_markup=reply_markup
        )

async def handle_notification_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle notification settings buttons"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if query.data == "disable_notifications":
        # Disable notifications
        if user_id in user_preferences:
            user_preferences[user_id]['notification_scheduled'] = False
            
            # Remove any pending jobs for this user
            current_jobs = context.job_queue.get_jobs_by_name(f"notification_{user_id}")
            for job in current_jobs:
                job.schedule_removal()
            
            await query.edit_message_text(
                "✅ Уведомления о новых квартирах отключены. "
                "Вы можете включить их снова после поиска квартиры через /seerent"
            )
    
    elif query.data == "enable_notifications":
        await query.edit_message_text(
            "Для включения уведомлений, найдите понравившуюся квартиру через /seerent, "
            "а затем нажмите '👍 Нравится'. После этого вы начнете получать уведомления "
            "о похожих квартирах."
        )
    
    elif query.data == "change_preferences":
        await query.edit_message_text(
            "Для изменения предпочтений начните поиск квартиры заново с помощью команды /seerent"
        )

def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Create conversation handler for rental viewing
    rental_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("seerent", seerent_command),
            CommandHandler("start", start),
            CallbackQueryHandler(handle_start_search, pattern="^start_search$")
        ],
        states={
            DISTRICT_SELECT: [
                CallbackQueryHandler(select_district, pattern="^district_\d+$"),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            VIEWING_LISTINGS: [
                CallbackQueryHandler(handle_listing_response, pattern="^(like|dislike|continue|change_district|stop|acknowledge_change)$")
            ]
        },
        fallbacks=[CallbackQueryHandler(handle_stop, pattern="^stop$")]
    )
    
    # Create conversation handler for price estimation
    price_estimation_handler = ConversationHandler(
        entry_points=[CommandHandler("estimate", start_price_estimation)],
        states={
            FLOOR_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_floor),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            TOTAL_FLOORS_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_total_floors),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            AREA_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_area),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            ROOMS_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_rooms),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            BATHROOM_INPUT: [
                CallbackQueryHandler(process_bathroom, pattern="^bathroom_\d+$"),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
        },
        fallbacks=[CallbackQueryHandler(handle_stop, pattern="^stop$")]
    )
    
    # Add handlers to the application
    app.add_handler(rental_conv_handler)
    app.add_handler(price_estimation_handler)
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("notifications", notification_command))
    app.add_handler(CallbackQueryHandler(handle_notification_settings, pattern="^(disable_notifications|enable_notifications|change_preferences)$"))
    app.add_handler(CallbackQueryHandler(handle_continue_search, pattern="^continue_search$"))
    
    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()