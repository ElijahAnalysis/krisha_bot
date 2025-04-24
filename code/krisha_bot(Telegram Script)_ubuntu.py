import os
import pandas as pd
import numpy as np
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
import logging
from datetime import datetime, timedelta
import random

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
DATA_PATH = "/root/krisha_bot/data/regular_scrapping/cleaned/almaty_apartments_cleaned.csv"
MODEL_PATH = "/root/krisha_bot/models/krisha_almaty_rental_kmeans29_pipeline.joblib"
PRICE_MODEL_PATH = "/root/krisha_bot/models/krisha_almaty_rental_stacking.joblib"

# Bot token - place your token directly here or use environment variables
TOKEN = "8160649571:AAHRFMGnqO3LcMpDzxKfIFDjitBQ3onhZCE"
# Alternatively, use environment variables (more secure):
# TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

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
        """Check if data needs to be reloaded (every 2 days at 07:00)"""
        now = datetime.now()
    
        # Check if it's 07:00 (with a 5-minute window)
        is_reload_time = now.hour == 7 and 0 <= now.minute < 5
    
        days_passed = 0
        if self.last_loaded is not None:
            days_passed = (now - self.last_loaded).days
    
        # Reload if never loaded or if 2+ days passed and it's reload time
        if self.last_loaded is None or (days_passed >= 2 and is_reload_time):
            logger.info("Reloading data and model at scheduled time (every 2 days at 07:00)...")
            self.load_data_and_model()
            self.last_loaded = now
    
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
        f"💰 Оцениваю ориентировочную стоимость аренды вашей квартиры, 5 простых вопросов!\n\n"
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

async def handle_listing_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user's response to a listing"""
    query = update.callback_query
    await query.answer()
    
    user_response = query.data
    
    if user_response == "stop":
        # User wants to stop the process
        return await handle_stop(update, context)
    
    if user_response == "change_district":
        # User wants to change district
        return await show_district_selection(update, context)
    
    elif user_response == "dislike":
        # User didn't like this listing, show another one
        return await show_listing(update, context)
    
    elif user_response == "like":
        # User liked the listing
        current_listing = context.user_data.get('current_listing')
        
        if current_listing:
            # Record the cluster the user liked
            context.user_data['preferred_cluster'] = current_listing.get('cluster')
            
            # Show the URL to the user
            listing_url = current_listing.get('url', 'URL не найден')
            
            try:
                await query.edit_message_text(
                    f"🎉 Отлично! Вот ссылка на это объявление: {listing_url}\n\n"
                    f"Теперь я буду показывать вам похожие квартиры. Хотите продолжить просмотр?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✨ Да, показать еще", callback_data="show_more")],
                        [InlineKeyboardButton("❌ Отменить поиск", callback_data="stop")]
                    ])
                )
            except Exception as e:
                logger.error(f"Error showing URL: {e}")
                # If editing fails, send a new message
                await query.message.reply_text(
                    f"🎉 Отлично! Вот ссылка на это объявление: {listing_url}\n\n"
                    f"Теперь я буду показывать вам похожие квартиры. Хотите продолжить просмотр?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✨ Да, показать еще", callback_data="show_more")],
                        [InlineKeyboardButton("❌ Отменить поиск", callback_data="stop")]
                    ])
                )
            
            return VIEWING_LISTINGS
    
    elif user_response == "show_more":
        # Show another listing (will be from preferred cluster if available)
        return await show_listing(update, context)
    
    # Default: continue showing listings
    return await show_listing(update, context)

# ---- Price Estimation Functions ----

async def estimate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the price estimation process"""
    # Clear any previous user data for estimation
    if 'estimation_data' in context.user_data:
        context.user_data['estimation_data'] = {}
    else:
        context.user_data['estimation_data'] = {}
    
    # Start by asking for floor number
    message = await update.message.reply_text(
        "💰 Давайте оценим примерную стоимость аренды вашей квартиры!\n\n"
        "Для начала, укажите этаж квартиры (например, 5):\n\n"
        "Чтобы отменить оценку, нажмите /cancel"
    )
    
    # Store the message ID for potential cancellation
    context.user_data['current_message_id'] = message.message_id
    
    return FLOOR_INPUT

# Add a cancel handler for text commands during estimation
async def cancel_estimation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the estimation process"""
    await update.message.reply_text(
        "✅ Оценка стоимости отменена. Вы можете использовать команды:\n"
        "/seerent - для поиска квартир\n"
        "/estimate - для оценки стоимости\n"
        "/help - для получения справки"
    )
    return ConversationHandler.END

async def floor_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle floor input and ask for total floors"""
    try:
        floor = int(update.message.text.strip())
        if floor <= 0:
            await update.message.reply_text("Этаж должен быть положительным числом. Попробуйте снова:")
            return FLOOR_INPUT
        
        # Store the floor in user data
        context.user_data['estimation_data']['floor'] = floor
        
        # Ask for total floors
        await update.message.reply_text(
            f"Вы указали {floor} этаж.\n"
            f"Теперь укажите общее количество этажей в доме (например, 9):\n\n"
            f"Чтобы отменить оценку, нажмите /cancel"
        )
        return TOTAL_FLOORS_INPUT
    
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное число для этажа (например, 5):"
        )
        return FLOOR_INPUT

async def total_floors_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle total floors input and ask for area"""
    try:
        total_floors = int(update.message.text.strip())
        if total_floors <= 0:
            await update.message.reply_text("Количество этажей должно быть положительным числом. Попробуйте снова:")
            return TOTAL_FLOORS_INPUT
        
        floor = context.user_data['estimation_data']['floor']
        if floor > total_floors:
            await update.message.reply_text(
                f"Этаж квартиры ({floor}) не может быть больше общего количества этажей ({total_floors}).\n"
                f"Пожалуйста, введите корректное значение:"
            )
            return TOTAL_FLOORS_INPUT
        
        # Store the total floors in user data
        context.user_data['estimation_data']['total_floors'] = total_floors
        
        # Ask for area
        await update.message.reply_text(
            f"Вы указали {total_floors} этажей всего.\n"
            f"Теперь укажите площадь квартиры в квадратных метрах (например, 45.5):\n\n"
            f"Чтобы отменить оценку, нажмите /cancel"
        )
        return AREA_INPUT
    
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное число для общего количества этажей (например, 9):"
        )
        return TOTAL_FLOORS_INPUT

async def area_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle area input and ask for rooms"""
    try:
        area = float(update.message.text.strip().replace(',', '.'))
        if area <= 0:
            await update.message.reply_text("Площадь должна быть положительным числом. Попробуйте снова:")
            return AREA_INPUT
        
        # Store the area in user data
        context.user_data['estimation_data']['area_sqm'] = area
        
        # Ask for number of rooms
        await update.message.reply_text(
            f"Вы указали площадь {area} м².\n"
            f"Теперь укажите количество жилых комнат (например, 2):\n\n"
            f"Чтобы отменить оценку, нажмите /cancel"
        )
        return ROOMS_INPUT
    
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное число для площади (например, 45.5):"
        )
        return AREA_INPUT

async def rooms_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle rooms input and ask for bathroom type"""
    try:
        rooms = int(update.message.text.strip())
        if rooms <= 0:
            await update.message.reply_text("Количество комнат должно быть положительным числом. Попробуйте снова:")
            return ROOMS_INPUT
        
        # Store the rooms in user data
        context.user_data['estimation_data']['rooms'] = rooms
        
        # Create keyboard for bathroom type selection with stop button
        keyboard = [
            [InlineKeyboardButton("Раздельный", callback_data="bathroom_2")],
            [InlineKeyboardButton("Совмещенный", callback_data="bathroom_5")],
            [InlineKeyboardButton("2 санузла или больше", callback_data="bathroom_0")],
            [InlineKeyboardButton("❌ Отменить оценку", callback_data="stop")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Вы указали {rooms} {_format_rooms_word(rooms)}.\n"
            f"Выберите тип санузла:",
            reply_markup=reply_markup
        )
        return BATHROOM_INPUT
    
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное число комнат (например, 2):"
        )
        return ROOMS_INPUT

def _format_rooms_word(count):
    """Format the word 'room' based on count in Russian"""
    if count % 10 == 1 and count % 100 != 11:
        return "комнату"
    elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
        return "комнаты"
    else:
        return "комнат"

async def bathroom_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle bathroom type selection and show price estimation"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "stop":
        return await handle_stop(update, context)
    
    # Extract bathroom code from callback data
    bathroom_code = int(query.data.split('_')[1])
    
    # Store the bathroom type in user data
    context.user_data['estimation_data']['bathroom_code'] = bathroom_code
    
    # Get all estimation data
    data = context.user_data['estimation_data']
    floor = data['floor']
    total_floors = data['total_floors']
    area_sqm = data['area_sqm']
    rooms = data['rooms']
    
    # Get bathroom description for display
    bathroom_description = BATHROOM_MAPPING.get(bathroom_code, "Неизвестно")
    
    # Estimate the price
    estimated_price = data_manager.estimate_price(floor, total_floors, area_sqm, rooms, bathroom_code)
    
    if estimated_price is None:
        await query.edit_message_text(
            "😕 К сожалению, не удалось оценить стоимость с указанными параметрами. "
            "Попробуйте другие значения или воспользуйтесь командой /estimate снова."
        )
    else:
        # Calculate price range (±10%)
        lower_price = int(estimated_price * 0.9)
        upper_price = int(estimated_price * 1.1)
        
        # Format prices with spaces
        lower_price_str = f"{lower_price:,}".replace(',', ' ')
        upper_price_str = f"{upper_price:,}".replace(',', ' ')
        price_str = f"{estimated_price:,}".replace(',', ' ')
        
        # Prepare result message with price interval
        message = (
            f"💰 <b>Оценка стоимости аренды</b>\n\n"
            f"На основе введенных данных:\n"
            f"🏢 Этаж: {floor}/{total_floors}\n"
            f"📐 Площадь: {area_sqm} м²\n"
            f"🚪 Комнат: {rooms}\n"
            f"🚽 Санузел: {bathroom_description}\n\n"
            f"<b>Примерная стоимость аренды:</b> {lower_price_str} - {upper_price_str} тенге в месяц\n"
            f"<i>(в среднем {price_str} тенге)</i>\n"
        )
        
        # Create keyboard for next actions
        keyboard = [
            [InlineKeyboardButton("🔄 Новая оценка", callback_data="new_estimate")],
            [InlineKeyboardButton("❌ Закончить", callback_data="stop")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    
    return ConversationHandler.END

async def handle_post_estimate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user's decision after price estimation"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "new_estimate":
        # User wants a new estimate
        await query.edit_message_text("Начинаем новую оценку...")
        # We need to create a new message since we can't transition back to text input from a callback
        await query.message.reply_text(
            "💰 Давайте оценим примерную стоимость аренды вашей квартиры!\n\n"
            "Для начала, укажите этаж квартиры (например, 5):\n\n"
            "Чтобы отменить оценку, нажмите /cancel"
        )
        return FLOOR_INPUT
    
    elif query.data == "stop":
        # User wants to stop
        return await handle_stop(update, context)
    
    return ConversationHandler.END

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Set up conversation handlers
    # Apartment viewing conversation
    apartment_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("seerent", seerent_command)],
        states={
            DISTRICT_SELECT: [
                CallbackQueryHandler(select_district, pattern=r"^district_\d+$"),
                CallbackQueryHandler(handle_stop, pattern="^stop$")
            ],
            VIEWING_LISTINGS: [
                CallbackQueryHandler(handle_listing_response)
            ],
        },
        fallbacks=[CommandHandler("cancel", handle_stop)]
    )
    
    # Price estimation conversation
    estimation_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("estimate", estimate_command)],
        states={
            FLOOR_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, floor_input)
            ],
            TOTAL_FLOORS_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, total_floors_input)
            ],
            AREA_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, area_input)
            ],
            ROOMS_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, rooms_input)
            ],
            BATHROOM_INPUT: [
                CallbackQueryHandler(bathroom_input)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_estimation)]
    )
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(apartment_conv_handler)
    application.add_handler(estimation_conv_handler)
    application.add_handler(CallbackQueryHandler(handle_post_estimate, pattern="^(start_search|new_estimate|stop)$"))
    
    # Log all errors
    application.add_error_handler(lambda update, context: logger.error(f"Update {update} caused error {context.error}"))
    
    logger.info("Starting bot...")
    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()