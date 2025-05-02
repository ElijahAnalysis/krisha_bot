# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import randint, uniform
import tensorflow as tf
from pathlib import Path
import re
from tensorflow.keras.models import load_model
from joblib import load

def directory_files(directory, item_type=None):
    """Get sorted list of files with specified extension from a directory"""
    return sorted([str(path) for path in directory.rglob(f'*.{item_type}')])

def load_images_with_ids(image_paths, target_size=(224, 224)):
    """Load images and associate them with listing IDs extracted from filenames"""
    loaded_images = {}

    for path in image_paths:
        try:
            # Extract listing ID using regex
            match = re.search(r'listing_(\d+)', path)
            if not match:
                print(f"Could not extract ID from path: {path}")
                continue

            listing_id = match.group(1)

            # Load and preprocess image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = image / 255.0
            image.set_shape((target_size[0], target_size[1], 3))

            # Store using listing ID
            loaded_images[listing_id] = image

        except Exception as e:
            print(f"Error loading image at {path}: {e}")

    return loaded_images

# Load the data and models 
file_path = "/root/krisha_bot/data/regular_scrapping/scrapped/almaty_apartments.csv"
images_path = "/root/krisha_bot/data/regular_scrapping/images/almaty_rental_images/images"
cnn_path = "/root/krisha_bot/models/krisha_image_mobilenetv3.keras"
clustering_path = "/root/krisha_bot/models/krisha_almaty_rental_kmeans25_ext.joblib"
scaler_path = "/root/krisha_bot/models/krisha_almaty_rental_tabular_sscaler.joblib"

# Load initial data
krisha_almaty_rental = pd.read_csv(file_path)

# Load models
cnn = load_model(cnn_path)
clustering = load(clustering_path)
scaler = load(scaler_path)

# Load images
images_directory = Path(images_path)
images_directory_files = directory_files(directory=images_directory, item_type='jpg')
images_directory_pictures = load_images_with_ids(image_paths=images_directory_files, target_size=(224, 224))

# Extract IDs and images
image_ids = list(images_directory_pictures.keys())
images = list(images_directory_pictures.values())

# Stack images into a batch for prediction
images_batch = tf.stack(images)

# STEP 1: Get CNN embeddings from the model
print("Generating CNN embeddings...")
cnn_embeddings = cnn.predict(images_batch)

# STEP 2: Create a DataFrame with the embeddings
cnn_embeddings_df = pd.DataFrame(
    data=cnn_embeddings,
    index=pd.Index(image_ids, name='id')
).reset_index()

# Convert ID to integer for merging
cnn_embeddings_df['id'] = cnn_embeddings_df['id'].astype('int')

# Data cleaning
print("Cleaning rental data...")
# Delete rows with missing values in floor and total_floors
krisha_almaty_rental = krisha_almaty_rental.dropna(subset=['floor', 'total_floors'])

# Fill missing values in categorical columns with 'неизвестно'
categorical_columns = ["bathroom", "furniture", "parking", "security", "balcony"]
for column in categorical_columns:
    krisha_almaty_rental[column] = krisha_almaty_rental[column].fillna('неизвестно')

# Define encoding mappings
encoding_mappings = {
    'bathroom': {
        'разделен': 2,
        'совмещен': 5,
        'неизвестно': 1,
        'разделен, совмещен': 3,
        'совмещенный': 6,
        '2 с/у и более': 0,
        'раздельный': 4
    },
    'furniture': {'неизвестно': 34, 'обеденный стол, шкаф для одежды, диван': 44, 'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 20, 'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол': 19, 'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, диван': 18, 'обеденный стол, шкаф для одежды, кухонный гарнитур, диван': 46, 'обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол': 47, 'обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 48, 'кровать, шкаф для одежды, кухонный гарнитур, диван': 26, 'кровать, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 28, 'обеденный стол, кухонный гарнитур, рабочий стол': 39, 'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур': 17, 'обеденный стол, кухонный гарнитур, диван': 38, 'кровать, обеденный стол': 7, 'шкаф для одежды, диван': 53, 'кровать, шкаф для одежды, кухонный гарнитур': 25, 'кровать, обеденный стол, кухонный гарнитур, рабочий стол': 11, 'кровать, обеденный стол, шкаф для одежды, рабочий стол, диван': 22, 'кровать, шкаф для одежды, рабочий стол, диван': 29, 'кухонный гарнитур, диван': 31, 'кровать': 1, 'кровать, обеденный стол, шкаф для одежды, диван': 16, 'кровать, шкаф для одежды, кухонный гарнитур, рабочий стол': 27, 'кровать, обеденный стол, кухонный гарнитур, диван': 10, 'шкаф для одежды, кухонный гарнитур': 54, 'шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 57, 'шкаф для одежды': 52, 'кухонный гарнитур': 30, 'кровать, обеденный стол, диван': 8, 'обеденный стол, шкаф для одежды, кухонный гарнитур': 45, 'диван': 0, 'кровать, обеденный стол, кухонный гарнитур, рабочий стол, диван': 12, 'обеденный стол, шкаф для одежды, рабочий стол, диван': 50, 'кровать, шкаф для одежды, диван': 24, 'шкаф для одежды, рабочий стол, диван': 58, 'кровать, обеденный стол, шкаф для одежды': 15, 'шкаф для одежды, кухонный гарнитур, диван': 55, 'кровать, кухонный гарнитур, диван': 4, 'кровать, обеденный стол, шкаф для одежды, рабочий стол': 21, 'обеденный стол': 35, 'обеденный стол, кухонный гарнитур': 37, 'обеденный стол, шкаф для одежды, рабочий стол': 49, 'кухонный гарнитур, рабочий стол, диван': 33, 'обеденный стол, кухонный гарнитур, рабочий стол, диван': 40, 'кровать, шкаф для одежды': 23, 'кровать, кухонный гарнитур, рабочий стол, диван': 6, 'кровать, кухонный гарнитур': 3, 'обеденный стол, рабочий стол, диван': 42, 'кровать, обеденный стол, кухонный гарнитур': 9, 'обеденный стол, шкаф для одежды': 43, 'шкаф для одежды, кухонный гарнитур, рабочий стол': 56, 'обеденный стол, рабочий стол': 41, 'кровать, диван': 2, 'обеденный стол, диван': 36, 'кровать, кухонный гарнитур, рабочий стол': 5, 'кровать, обеденный стол, рабочий стол': 13, 'кровать, обеденный стол, рабочий стол, диван': 14, 'кухонный гарнитур, рабочий стол': 32},
    'parking': {'неизвестно': 1, 'рядом охраняемая стоянка': 3, 'паркинг': 2, 'гараж': 0},
    'security': {'неизвестно': 32, 'охрана, домофон, видеонаблюдение': 41, 'домофон, видеонаблюдение': 8, 'домофон': 5, 'охрана, видеонаблюдение, видеодомофон': 35, 'охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 49, 'домофон, видеонаблюдение, видеодомофон': 9, 'решетки на окнах, домофон, кодовый замок, видеонаблюдение': 83, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 55, 'домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 19, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение': 54, 'охрана, домофон, сигнализация, видеонаблюдение': 61, 'охрана, домофон, кодовый замок, видеонаблюдение': 47, 'охрана, домофон, видеонаблюдение, видеодомофон, консьерж': 43, 'решетки на окнах, охрана, домофон, видеонаблюдение': 92, 'охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон': 48, 'решетки на окнах, охрана, домофон, видеонаблюдение, консьерж': 95, 'охрана, домофон, видеонаблюдение, видеодомофон': 42, 'домофон, кодовый замок, видеонаблюдение, видеодомофон': 14, 'домофон, кодовый замок': 11, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 56, 'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, видеодомофон, консьерж': 105, 'охрана, домофон': 38, 'домофон, кодовый замок, видеонаблюдение': 13, 'охрана, домофон, видеонаблюдение, консьерж': 44, 'решетки на окнах, охрана, домофон, кодовый замок, видеодомофон': 96, 'видеодомофон, консьерж': 1, 'решетки на окнах, домофон': 79, 'домофон, сигнализация, видеонаблюдение': 24, 'решетки на окнах, охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон': 98, 'охрана, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 69, 'охрана, домофон, сигнализация, видеонаблюдение, видеодомофон': 62, 'решетки на окнах, охрана, домофон': 91, 'охрана, кодовый замок, видеонаблюдение': 67, 'решетки на окнах, охрана, видеонаблюдение': 90, 'видеонаблюдение': 2, 'охрана, домофон, видеодомофон': 39, 'охрана, домофон, консьерж': 58, 'охрана, видеонаблюдение': 34, 'охрана': 33, 'домофон, видеодомофон': 6, 'охрана, домофон, сигнализация': 59, 'охрана, сигнализация, видеонаблюдение, видеодомофон': 76, 'решетки на окнах': 77, 'охрана, кодовый замок, видеонаблюдение, видеодомофон': 68, 'домофон, кодовый замок, сигнализация, видеодомофон': 17, 'охрана, домофон, видеодомофон, консьерж': 40, 'решетки на окнах, домофон, кодовый замок, сигнализация, видеонаблюдение': 85, 'охрана, кодовый замок, видеодомофон': 66, 'охрана, сигнализация, видеонаблюдение': 75, 'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 102, 'охрана, видеонаблюдение, консьерж': 37, 'кодовый замок, видеонаблюдение': 28, 'видеонаблюдение, видеодомофон': 3, 'консьерж': 31, 'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 101, 'решетки на окнах, охрана, консьерж': 111, 'охрана, домофон, кодовый замок, сигнализация, видеодомофон': 52, 'домофон, кодовый замок, видеодомофон': 12, 'домофон, видеонаблюдение, консьерж': 10, 'решетки на окнах, кодовый замок, видеонаблюдение': 89, 'сигнализация, видеодомофон': 114, 'кодовый замок, видеонаблюдение, видеодомофон': 29, 'охрана, домофон, сигнализация, видеонаблюдение, консьерж': 64, 'решетки на окнах, охрана, сигнализация, видеонаблюдение, видеодомофон, консьерж': 112, 'домофон, видеодомофон, консьерж': 7, 'кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 30, 'решетки на окнах, домофон, видеонаблюдение': 80, 'охрана, домофон, кодовый замок': 45, 'решетки на окнах, охрана, кодовый замок, видеодомофон, консьерж': 107, 'домофон, сигнализация, видеодомофон': 23, 'решетки на окнах, охрана, кодовый замок, видеонаблюдение': 108, 'домофон, консьерж': 21, 'охрана, кодовый замок': 65, 'решетки на окнах, домофон, кодовый замок': 82, 'решетки на окнах, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 86, 'домофон, сигнализация': 22, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, консьерж': 57, 'решетки на окнах, охрана, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 109, 'решетки на окнах, охрана, домофон, кодовый замок, видеонаблюдение': 97, 'охрана, домофон, кодовый замок, видеодомофон': 46, 'охрана, домофон, кодовый замок, видеонаблюдение, консьерж': 50, 'кодовый замок': 25, 'домофон, кодовый замок, сигнализация, видеонаблюдение': 18, 'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, консьерж': 106, 'видеодомофон': 0, 'охрана, кодовый замок, видеонаблюдение, консьерж': 70, 'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение': 100, 'решетки на окнах, домофон, сигнализация, видеонаблюдение, видеодомофон': 88, 'домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 20, 'кодовый замок, видеодомофон': 26, 'охрана, кодовый замок, консьерж': 71, 'охрана, домофон, кодовый замок, консьерж': 51, 'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, видеодомофон': 104, 'решетки на окнах, домофон, видеонаблюдение, видеодомофон': 81, 'кодовый замок, видеодомофон, консьерж': 27, 'охрана, видеонаблюдение, видеодомофон, консьерж': 36, 'решетки на окнах, домофон, сигнализация, видеонаблюдение': 87, 'охрана, домофон, сигнализация, видеонаблюдение, видеодомофон, консьерж': 63, 'охрана, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 73, 'решетки на окнах, охрана, домофон, консьерж': 103, 'решетки на окнах, домофон, кодовый замок, видеонаблюдение, видеодомофон': 84, 'домофон, кодовый замок, видеонаблюдение, консьерж': 16, 'решетки на окнах, охрана, домофон, видеонаблюдение, видеодомофон, консьерж': 94, 'охрана, кодовый замок, сигнализация, видеонаблюдение': 72, 'решетки на окнах, видеонаблюдение': 78},
}

# Updated full_address mapping
full_address_mapping = {
    'Алматы, Алмалинский р-н\nпоказать на карте': 2,
    'Алматы, Бостандыкский р-н\nпоказать на карте': 4,
    'Алматы, Наурызбайский р-н\nпоказать на карте': 7,
    'Алматы\nпоказать на карте': 0,
    'Алматы, Ауэзовский р-н\nпоказать на карте': 3,
    'Алматы, Медеуский р-н\nпоказать на карте': 6,
    'Алматы, Алатауский р-н\nпоказать на карте': 1,
    'Алматы, Турксибский р-н\nпоказать на карте': 8,
    'Алматы, Жетысуский р-н\nпоказать на карте': 5
}

# Create encoded columns
krisha_almaty_rental['full_address_code'] = krisha_almaty_rental['full_address'].map(full_address_mapping)
krisha_almaty_rental['furniture_code'] = krisha_almaty_rental['furniture'].map(encoding_mappings['furniture'])
krisha_almaty_rental['parking_code'] = krisha_almaty_rental['parking'].map(encoding_mappings['parking'])
krisha_almaty_rental['security_code'] = krisha_almaty_rental['security'].map(encoding_mappings['security'])
krisha_almaty_rental['bathroom_code'] = krisha_almaty_rental['bathroom'].map(encoding_mappings['bathroom'])

# Define tabular columns for scaling and clustering
tabular_columns = [
    'floor', 'total_floors', 'area_sqm', 'rooms', 'price',
    'full_address_code', 'furniture_code', 'parking_code', 'security_code',
    'bathroom_code'
]

# Scale the tabular features
print("Scaling tabular features...")

# Step 1: Drop NaN values
krisha_almaty_rental_cleaned = krisha_almaty_rental.dropna()

# Step 2: Scale tabular columns for clustering
krisha_almaty_rental_tabular_scaled = scaler.transform(krisha_almaty_rental_cleaned[tabular_columns])

# STEP 3: Merge the tabular data with CNN embeddings
print("Merging tabular data with CNN embeddings...")
essential_columns = [
    'floor', 'total_floors', 'area_sqm', 'rooms', 'price',
    'full_address_code', 'furniture_code', 'parking_code', 'security_code',
    'bathroom_code', 'contact_name', 'title', 'url', 'id'
]
krisha_almaty_rental_subset = krisha_almaty_rental_cleaned[essential_columns].dropna()

# Merge with CNN embeddings
krisha_almaty_rental_ext = pd.merge(krisha_almaty_rental_subset, cnn_embeddings_df, on='id')

# Identify embedding columns (they'll be numeric columns from CNN output)
embedding_columns = [col for col in cnn_embeddings_df.columns if col != 'id']

print("Generating cluster predictions using scaled data...")

# STEP 4: Generate cluster predictions using scaled data
print("Generating cluster predictions using scaled data...")

# Create a DataFrame with both tabular scaled data and CNN embeddings
# First, we need to make sure we have the same set of records in both datasets
krisha_almaty_rental_ext_ids = krisha_almaty_rental_ext['id'].values
krisha_almaty_rental_cleaned_ids = krisha_almaty_rental_cleaned['id'].values

# Find common IDs
common_ids = np.intersect1d(krisha_almaty_rental_ext_ids, krisha_almaty_rental_cleaned_ids)

# Filter both datasets to only include common IDs
krisha_almaty_rental_ext_filtered = krisha_almaty_rental_ext[krisha_almaty_rental_ext['id'].isin(common_ids)]
krisha_almaty_rental_cleaned_filtered = krisha_almaty_rental_cleaned[krisha_almaty_rental_cleaned['id'].isin(common_ids)]

# Ensure they are in the same order
krisha_almaty_rental_ext_filtered = krisha_almaty_rental_ext_filtered.sort_values('id')
krisha_almaty_rental_cleaned_filtered = krisha_almaty_rental_cleaned_filtered.sort_values('id')

# Get the scaled tabular data for the common IDs
tabular_indices = [list(krisha_almaty_rental_cleaned['id']).index(id_) for id_ in common_ids]
tabular_scaled_filtered = krisha_almaty_rental_tabular_scaled[tabular_indices]

# Create a DataFrame from the scaled tabular data
tabular_scaled_df = pd.DataFrame(
    tabular_scaled_filtered,
    columns=tabular_columns
)

# Get the CNN embeddings for the common IDs
cnn_embeddings_filtered = krisha_almaty_rental_ext_filtered[embedding_columns]

# Combine tabular scaled data with CNN embeddings
clustering_features_scaled = pd.concat([tabular_scaled_df, cnn_embeddings_filtered], axis=1)

# Convert all column names to strings to avoid feature name type issues
clustering_features_scaled.columns = clustering_features_scaled.columns.astype(str)

# Check for NaN or infinite values and handle them
if clustering_features_scaled.isnull().sum().sum() > 0 or np.isinf(clustering_features_scaled.values).sum() > 0:
    print("There are NaN or infinite values in the features.")
    clustering_features_scaled = clustering_features_scaled.fillna(0).replace([np.inf, -np.inf], 0)

# Predict clusters using the scaled data
clusters = clustering.predict(clustering_features_scaled)

# Add cluster assignments to the filtered dataframe
krisha_almaty_rental_ext_filtered['cluster'] = clusters

# Now merge this back to the original dataframe
krisha_almaty_rental_ext = pd.merge(
    krisha_almaty_rental_ext.drop('cluster', errors='ignore'), 
    krisha_almaty_rental_ext_filtered[['id', 'cluster']], 
    on='id', 
    how='left'
)

# STEP 5: Combine original (unscaled) data with clusters
# Keep essential columns along with the cluster column (including original data)
columns_to_keep = ['floor', 'total_floors', 'area_sqm', 'rooms', 'price',
                   'full_address_code', 'furniture_code', 'parking_code', 'security_code',
                   'bathroom_code', 'contact_name', 'title', 'url', 'id', 'cluster']

# Save the combined data (original + cluster) to a CSV file
output_path = "/root/krisha_bot/data/regular_scrapping/cleaned/almaty_apartments_clustered.csv"
krisha_almaty_rental_ext[columns_to_keep].to_csv(output_path, index=False)

print(f'CSV file with {krisha_almaty_rental_ext[columns_to_keep].shape} shape saved to {output_path}')