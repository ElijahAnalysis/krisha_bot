import pandas as pd
import numpy as np
from joblib import load

# Load the data and model
file_path = "/root/krisha_bot/data/regular_scrapping/scrapped/almaty_apartments.csv"
clustering_model_path = "/root/krisha_bot/models/krisha_almaty_rental_kmeans26_pipeline.joblib"

krisha_almaty_rental = pd.read_csv(file_path)
clustering_model = load(clustering_model_path)

print(f'Initial data shape: {krisha_almaty_rental.shape}')

# Step 1: Delete rows with missing values in floor and total_floors
krisha_almaty_rental = krisha_almaty_rental.dropna(subset=['floor', 'total_floors'])
print(f'After dropping floor/total_floors NaN: {krisha_almaty_rental.shape}')

# Step 2: Fill missing values in categorical columns with 'неизвестно'
categorical_columns = ["bathroom", "furniture", "parking", "security", "balcony"]
for column in categorical_columns:
    if column in krisha_almaty_rental.columns:
        krisha_almaty_rental[column] = krisha_almaty_rental[column].fillna('неизвестно')

# Step 3: Define encoding mappings with updated values
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
    'furniture': {
        'неизвестно': 34, 'обеденный стол, шкаф для одежды, диван': 44, 
        'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 20, 
        'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол': 19, 
        'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур, диван': 18, 
        'обеденный стол, шкаф для одежды, кухонный гарнитур, диван': 46, 
        'обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол': 47, 
        'обеденный стол, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 48, 
        'кровать, шкаф для одежды, кухонный гарнитур, диван': 26, 
        'кровать, шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 28, 
        'обеденный стол, кухонный гарнитур, рабочий стол': 39, 
        'кровать, обеденный стол, шкаф для одежды, кухонный гарнитур': 17, 
        'обеденный стол, кухонный гарнитур, диван': 38, 'кровать, обеденный стол': 7, 
        'шкаф для одежды, диван': 53, 'кровать, шкаф для одежды, кухонный гарнитур': 25, 
        'кровать, обеденный стол, кухонный гарнитур, рабочий стол': 11, 
        'кровать, обеденный стол, шкаф для одежды, рабочий стол, диван': 22, 
        'кровать, шкаф для одежды, рабочий стол, диван': 29, 'кухонный гарнитур, диван': 31, 
        'кровать': 1, 'кровать, обеденный стол, шкаф для одежды, диван': 16, 
        'кровать, шкаф для одежды, кухонный гарнитур, рабочий стол': 27, 
        'кровать, обеденный стол, кухонный гарнитур, диван': 10, 'шкаф для одежды, кухонный гарнитур': 54, 
        'шкаф для одежды, кухонный гарнитур, рабочий стол, диван': 57, 'шкаф для одежды': 52, 
        'кухонный гарнитур': 30, 'кровать, обеденный стол, диван': 8, 
        'обеденный стол, шкаф для одежды, кухонный гарнитур': 45, 'диван': 0, 
        'кровать, обеденный стол, кухонный гарнитур, рабочий стол, диван': 12, 
        'обеденный стол, шкаф для одежды, рабочий стол, диван': 50, 'кровать, шкаф для одежды, диван': 24, 
        'шкаф для одежды, рабочий стол, диван': 58, 'кровать, обеденный стол, шкаф для одежды': 15, 
        'шкаф для одежды, кухонный гарнитур, диван': 55, 'кровать, кухонный гарнитур, диван': 4, 
        'кровать, обеденный стол, шкаф для одежды, рабочий стол': 21, 'обеденный стол': 35, 
        'обеденный стол, кухонный гарнитур': 37, 'обеденный стол, шкаф для одежды, рабочий стол': 49, 
        'кухонный гарнитур, рабочий стол, диван': 33, 'обеденный стол, кухонный гарнитур, рабочий стол, диван': 40, 
        'кровать, шкаф для одежды': 23, 'кровать, кухонный гарнитур, рабочий стол, диван': 6, 
        'кровать, кухонный гарнитур': 3, 'обеденный стол, рабочий стол, диван': 42, 
        'кровать, обеденный стол, кухонный гарнитур': 9, 'обеденный стол, шкаф для одежды': 43, 
        'шкаф для одежды, кухонный гарнитур, рабочий стол': 56, 'обеденный стол, рабочий стол': 41, 
        'кровать, диван': 2, 'обеденный стол, диван': 36, 'кровать, кухонный гарнитур, рабочий стол': 5, 
        'кровать, обеденный стол, рабочий стол': 13, 'кровать, обеденный стол, рабочий стол, диван': 14, 
        'кухонный гарнитур, рабочий стол': 32,
        # Add the missing furniture values found in diagnostic
        'кровать, рабочий стол': 59,
        'кровать, шкаф для одежды, рабочий стол': 60,
        'рабочий стол': 61,
        'рабочий стол, диван': 62,
        'шкаф для одежды, рабочий стол': 63
    },
    'parking': {
        'неизвестно': 1, 
        'рядом охраняемая стоянка': 3, 
        'паркинг': 2, 
        'гараж': 0
    },
    'security': {
        'неизвестно': 32, 'охрана, домофон, видеонаблюдение': 41, 'домофон, видеонаблюдение': 8, 
        'домофон': 5, 'охрана, видеонаблюдение, видеодомофон': 35, 
        'охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 49, 
        'домофон, видеонаблюдение, видеодомофон': 9, 'решетки на окнах, домофон, кодовый замок, видеонаблюдение': 83, 
        'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 55, 
        'домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 19, 
        'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение': 54, 
        'охрана, домофон, сигнализация, видеонаблюдение': 61, 'охрана, домофон, кодовый замок, видеонаблюдение': 47, 
        'охрана, домофон, видеонаблюдение, видеодомофон, консьерж': 43, 
        'решетки на окнах, охрана, домофон, видеонаблюдение': 92, 
        'охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон': 48, 
        'решетки на окнах, охрана, домофон, видеонаблюдение, консьерж': 95, 
        'охрана, домофон, видеонаблюдение, видеодомофон': 42, 'домофон, кодовый замок, видеонаблюдение, видеодомофон': 14, 
        'домофон, кодовый замок': 11, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 56, 
        'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, видеодомофон, консьерж': 105, 
        'охрана, домофон': 38, 'домофон, кодовый замок, видеонаблюдение': 13, 'охрана, домофон, видеонаблюдение, консьерж': 44, 
        'решетки на окнах, охрана, домофон, кодовый замок, видеодомофон': 96, 'видеодомофон, консьерж': 1, 
        'решетки на окнах, домофон': 79, 'домофон, сигнализация, видеонаблюдение': 24, 
        'решетки на окнах, охрана, домофон, кодовый замок, видеонаблюдение, видеодомофон': 98, 
        'охрана, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 69, 
        'охрана, домофон, сигнализация, видеонаблюдение, видеодомофон': 62, 'решетки на окнах, охрана, домофон': 91, 
        'охрана, кодовый замок, видеонаблюдение': 67, 'решетки на окнах, охрана, видеонаблюдение': 90, 
        'видеонаблюдение': 2, 'охрана, домофон, видеодомофон': 39, 'охрана, домофон, консьерж': 58, 
        'охрана, видеонаблюдение': 34, 'охрана': 33, 'домофон, видеодомофон': 6, 'охрана, домофон, сигнализация': 59, 
        'охрана, сигнализация, видеонаблюдение, видеодомофон': 76, 'решетки на окнах': 77, 
        'охрана, кодовый замок, видеонаблюдение, видеодомофон': 68, 'домофон, кодовый замок, сигнализация, видеодомофон': 17, 
        'охрана, домофон, видеодомофон, консьерж': 40, 'решетки на окнах, домофон, кодовый замок, сигнализация, видеонаблюдение': 85, 
        'охрана, кодовый замок, видеодомофон': 66, 'охрана, сигнализация, видеонаблюдение': 75, 
        'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 102, 
        'охрана, видеонаблюдение, консьерж': 37, 'кодовый замок, видеонаблюдение': 28, 'видеонаблюдение, видеодомофон': 3, 
        'консьерж': 31, 'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 101, 
        'решетки на окнах, охрана, консьерж': 111, 'охрана, домофон, кодовый замок, сигнализация, видеодомофон': 52, 
        'домофон, кодовый замок, видеодомофон': 12, 'домофон, видеонаблюдение, консьерж': 10, 
        'решетки на окнах, кодовый замок, видеонаблюдение': 89, 'сигнализация, видеодомофон': 114, 
        'кодовый замок, видеонаблюдение, видеодомофон': 29, 'охрана, домофон, сигнализация, видеонаблюдение, консьерж': 64, 
        'решетки на окнах, охрана, сигнализация, видеонаблюдение, видеодомофон, консьерж': 112, 
        'домофон, видеодомофон, консьерж': 7, 'кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 30, 
        'решетки на окнах, домофон, видеонаблюдение': 80, 'охрана, домофон, кодовый замок': 45, 
        'решетки на окнах, охрана, кодовый замок, видеодомофон, консьерж': 107, 'домофон, сигнализация, видеодомофон': 23, 
        'решетки на окнах, охрана, кодовый замок, видеонаблюдение': 108, 'домофон, консьерж': 21, 
        'охрана, кодовый замок': 65, 'решетки на окнах, домофон, кодовый замок': 82, 
        'решетки на окнах, домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 86, 
        'домофон, сигнализация': 22, 'охрана, домофон, кодовый замок, сигнализация, видеонаблюдение, консьерж': 57, 
        'решетки на окнах, охрана, кодовый замок, видеонаблюдение, видеодомофон, консьерж': 109, 
        'решетки на окнах, охрана, домофон, кодовый замок, видеонаблюдение': 97, 'охрана, домофон, кодовый замок, видеодомофон': 46, 
        'охрана, домофон, кодовый замок, видеонаблюдение, консьерж': 50, 'кодовый замок': 25, 
        'домофон, кодовый замок, сигнализация, видеонаблюдение': 18, 'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, консьерж': 106, 
        'видеодомофон': 0, 'охрана, кодовый замок, видеонаблюдение, консьерж': 70, 
        'решетки на окнах, охрана, домофон, кодовый замок, сигнализация, видеонаблюдение': 100, 
        'решетки на окнах, домофон, сигнализация, видеонаблюдение, видеодомофон': 88, 
        'домофон, кодовый замок, сигнализация, видеонаблюдение, видеодомофон, консьерж': 20, 'кодовый замок, видеодомофон': 26, 
        'охрана, кодовый замок, консьерж': 71, 'охрана, домофон, кодовый замок, консьерж': 51, 
        'решетки на окнах, охрана, домофон, сигнализация, видеонаблюдение, видеодомофон': 104, 
        'решетки на окнах, домофон, видеонаблюдение, видеодомофон': 81, 'кодовый замок, видеодомофон, консьерж': 27, 
        'охрана, видеонаблюдение, видеодомофон, консьерж': 36, 'решетки на окнах, домофон, сигнализация, видеонаблюдение': 87, 
        'охрана, домофон, сигнализация, видеонаблюдение, видеодомофон, консьерж': 63, 
        'охрана, кодовый замок, сигнализация, видеонаблюдение, видеодомофон': 73, 'решетки на окнах, охрана, домофон, консьерж': 103, 
        'решетки на окнах, домофон, кодовый замок, видеонаблюдение, видеодомофон': 84, 
        'домофон, кодовый замок, видеонаблюдение, консьерж': 16, 'решетки на окнах, охрана, домофон, видеонаблюдение, видеодомофон, консьерж': 94, 
        'охрана, кодовый замок, сигнализация, видеонаблюдение': 72, 'решетки на окнах, видеонаблюдение': 78
    }
}

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

# Step 4: Function to safely map values with fallback
def safe_map_with_fallback(series, mapping, fallback_value=None):
    """
    Safely map values, assigning new codes to unmapped values instead of NaN
    """
    if fallback_value is None:
        fallback_value = max(mapping.values()) + 1 if mapping else 0
    
    # Get unique values not in mapping
    unique_values = series.unique()
    unmapped_values = [val for val in unique_values if val not in mapping]
    
    # Create extended mapping with new codes for unmapped values
    extended_mapping = mapping.copy()
    current_max = max(mapping.values()) if mapping else -1
    
    for i, unmapped_val in enumerate(unmapped_values):
        extended_mapping[unmapped_val] = current_max + 1 + i
        print(f"  Assigning new code {current_max + 1 + i} to unmapped value: '{unmapped_val}'")
    
    return series.map(extended_mapping)

# Step 5: Apply mappings with fallbacks
print("\nApplying robust mappings:")

print("Mapping full_address...")
krisha_almaty_rental['full_address_code'] = safe_map_with_fallback(
    krisha_almaty_rental['full_address'], full_address_mapping
)

print("Mapping furniture...")
krisha_almaty_rental['furniture_code'] = safe_map_with_fallback(
    krisha_almaty_rental['furniture'], encoding_mappings['furniture']
)

print("Mapping parking...")
krisha_almaty_rental['parking_code'] = safe_map_with_fallback(
    krisha_almaty_rental['parking'], encoding_mappings['parking']
)

print("Mapping security...")
krisha_almaty_rental['security_code'] = safe_map_with_fallback(
    krisha_almaty_rental['security'], encoding_mappings['security']
)

print("Mapping bathroom...")
krisha_almaty_rental['bathroom_code'] = safe_map_with_fallback(
    krisha_almaty_rental['bathroom'], encoding_mappings['bathroom']
)

print(f'\nAfter encoding (before clustering): {krisha_almaty_rental.shape}')

# Check for any remaining NaN values in encoded columns
encoded_columns = ['full_address_code', 'furniture_code', 'parking_code', 'security_code', 'bathroom_code']
for col in encoded_columns:
    nan_count = krisha_almaty_rental[col].isnull().sum()
    if nan_count > 0:
        print(f"WARNING: {col} still has {nan_count} NaN values!")

# Step 6: Apply clustering
feature_columns = ['floor', 'total_floors', 'area_sqm', 'rooms', 'price',
                   'full_address_code', 'furniture_code', 'parking_code', 
                   'security_code', 'bathroom_code']

# Check if we have all required columns
missing_cols = [col for col in feature_columns if col not in krisha_almaty_rental.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
else:
    # Only drop rows that have NaN in the clustering feature columns
    before_clustering_dropna = len(krisha_almaty_rental)
    krisha_almaty_rental = krisha_almaty_rental.dropna(subset=feature_columns)
    after_clustering_dropna = len(krisha_almaty_rental)
    
    print(f'After dropping NaN in clustering features: {krisha_almaty_rental.shape}')
    if before_clustering_dropna != after_clustering_dropna:
        print(f'Dropped {before_clustering_dropna - after_clustering_dropna} rows with NaN in clustering features')
    
    # Apply clustering
    krisha_almaty_rental['cluster'] = clustering_model.predict(krisha_almaty_rental[feature_columns])
    print(f'After clustering: {krisha_almaty_rental.shape}')

# Step 7: Keep only selected columns
columns_to_keep = ['floor', 'total_floors', 'area_sqm', 'rooms', 'price',
                   'full_address_code', 'furniture_code', 'parking_code', 'security_code',
                   'bathroom_code', 'contact_name', 'title', 'url', 'id', 'cluster']

# Check which columns exist
available_columns = [col for col in columns_to_keep if col in krisha_almaty_rental.columns]
missing_columns = [col for col in columns_to_keep if col not in krisha_almaty_rental.columns]

if missing_columns:
    print(f"WARNING: Missing columns in final selection: {missing_columns}")
    print(f"Available columns: {available_columns}")

krisha_almaty_rental = krisha_almaty_rental[available_columns]

# Final cleanup - only drop NaN for essential columns
essential_columns = ['floor', 'total_floors', 'area_sqm', 'rooms', 'price', 'cluster']
before_final_dropna = len(krisha_almaty_rental)
krisha_almaty_rental = krisha_almaty_rental.dropna(subset=essential_columns)
after_final_dropna = len(krisha_almaty_rental)

print(f'Final dataset shape: {krisha_almaty_rental.shape}')
if before_final_dropna != after_final_dropna:
    print(f'Final cleanup dropped {before_final_dropna - after_final_dropna} rows')

# Step 8: Save the cleaned dataset
output_path = r"C:\Users\User\Desktop\DATA SCIENCE\Github\krisha_bot\data\regular_scrapping\cleaned\almaty_apartments_clustered.csv"
krisha_almaty_rental.to_csv(output_path, index=False)


print(f"FINAL SUMMARY:")
print(f"Final cleaned data: {len(krisha_almaty_rental):,} rows")
print(f"Data saved to: {output_path}")
