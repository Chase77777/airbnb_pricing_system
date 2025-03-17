import pandas as pd 
import numpy as np
import re
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def clean_city(df):
    # Drop columns with all NaN values
    df_no_nan = df.dropna(axis=1, how='all')

    # in City column, rename
    df_no_nan['city'] = df_no_nan['city'].str.strip().str.replace('Bernal Heights, San Francisco', 'San Francisco', case=False)
    df_no_nan['city'] = df_no_nan['city'].str.replace('Noe Valley - San Francisco', 'San Francisco', case=False)
    df_no_nan['city'] = df_no_nan['city'].str.replace('San Francisco, Hayes Valley', 'San Francisco', case=False)
    df_no_nan['city'] = df_no_nan['city'].str.replace('South San Francisco', 'San Francisco', case=False)

    df_listings = pd.get_dummies(df_no_nan, columns=['city'], drop_first=False, prefix='city')

    return df_listings

def sklearn_ordinal_encoder(df, columns_to_encode):
    """
    Applies ordinal encoding to multiple specified columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_encode (list): List of column names to ordinal encode
        
    Returns:
        pd.DataFrame: DataFrame with encoded columns added
    """
    # Create a copy to avoid modifying original DataFrame
    df_encoded = df.copy()
    
    # Dictionary to store category orders for verification
    category_orders = {}

    for col in columns_to_encode:
        # Check if column exists in DataFrame
        if col not in df_encoded.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        # Get unique categories while preserving order (exclude NaNs)
        unique_cats = df_encoded[col].dropna().unique().tolist()
        
        # Skip if no valid categories found
        if len(unique_cats) == 0:
            print(f"⚠️  No valid categories found for {col}, skipping encoding")
            continue

        # Store category order for verification
        category_orders[col] = unique_cats

        # Create categorical type with detected order
        df_encoded[col] = pd.Categorical(df_encoded[col], 
                                       categories=unique_cats,
                                       ordered=True)

        # Create new column name for encoded values
        encoded_col_name = f"{col}_encoded"

        # Initialize and fit encoder
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                        unknown_value=-1)
        
        # Encode and handle missing values
        df_encoded[encoded_col_name] = ordinal_encoder.fit_transform(
            df_encoded[[col]]
        ).astype(int)

        print(f"\n✅ Ordinal encoding applied to {col}:")
        print(df_encoded[[col, encoded_col_name]].head())

    print("\n🔍 Category orders used for encoding:")
    for col, order in category_orders.items():
        print(f"{col}: {order}")

    return df_encoded

def clean_street(df_listings):
    """
    Cleans the 'street' column in-place within the DataFrame
    
    1. Splits street addresses at the first comma
    2. Removes 'CA' values by converting them to NaN
    3. Adds cleaned street column directly to original DataFrame
    """
    # Create copy to avoid SettingWithCopyWarning
    df = df_listings.copy()
    
    # Clean street addresses
    df['street'] = (
        df['street']
        .str.split(',', n=1, expand=True)[0]  # Split at first comma
        .str.strip()  # Remove whitespace
        .replace('CA', np.nan)  # Replace CA with NaN
    )
    
    # Optional verification
    print("Cleaned street values:")
    print(df['street'].unique())
    
    return df
 
def clean_zipcode(df_listings):
    """
    Cleans and encodes the 'zipcode' column in the DataFrame
    
    1. Removes 'CA ' prefix from zipcodes
    2. Replaces invalid 'CA' entries with NaN
    3. Encodes zipcodes using ordinal encoding
    4. Adds cleaned zipcode and encoded version to DataFrame
    """
    # Create copy to avoid SettingWithCopyWarning
    df = df_listings.copy()
    
    # Clean zipcode values
    df['zipcode'] = (
        df['zipcode']
        .str.replace('CA ', '', regex=False)  # Remove CA prefix
        .replace('CA', np.nan)  # Replace standalone CA with NaN
    )
    
    # Convert to string and handle missing values
    df['zipcode'] = df['zipcode'].astype(str).fillna('Unknown')
    
    # Apply ordinal encoding
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df['zipcode_encoded'] = ordinal_encoder.fit_transform(df[['zipcode']])
    
    # Verification
    print("✅ Ordinal encoding applied to zipcode:")
    print(df[['zipcode', 'zipcode_encoded']].head())
    print("\n🔍 Unique zipcodes after cleaning:", df['zipcode'].unique())
    
    return df

def clean_cancellation_policy(df):
    """
    Processes the cancellation policy column by:
    1. Categorizing specific policies into strictness groups
    2. Applying ordinal encoding based on strictness level
    3. Returning DataFrame with original and encoded columns
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Define categorization mapping
    def categorize_policy(policy):
        if policy in ['flexible', 'moderate']:
            return 'Lenient'
        elif policy in ['strict', 'strict_14_with_grace_period']:
            return 'Standard Strictness'
        elif policy in ['super_strict_30', 'super_strict_60']:
            return 'High Strictness'
        else:
            return pd.NA  # Use pandas NA for missing values

    # Apply categorization
    df_clean['cancellation_policy'] = df_clean['cancellation_policy'].apply(categorize_policy)
    
    # Define and validate ordinal order
    cancellation_order = ['Lenient', 'Standard Strictness', 'High Strictness']
    df_clean['cancellation_policy'] = pd.Categorical(
        df_clean['cancellation_policy'],
        categories=cancellation_order,
        ordered=True
    )

    # Apply ordinal encoding with proper NA handling
    ordinal_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    df_clean['cancellation_policy_encoded'] = ordinal_encoder.fit_transform(
        df_clean[['cancellation_policy']]
    )

    # Verification
    print("🔄 Unique policy categories:", df_clean['cancellation_policy'].unique())
    print("\n✅ Encoded values preview:")
    print(df_clean[['cancellation_policy', 'cancellation_policy_encoded']].head())
    
    return df_clean

def clean_date_columns(df,date_columns):
    
    for column in date_columns:
        df[column] = pd.to_datetime(df[column])

    return df

def clean_price_columns(df,price_columns):

    for column in price_columns:
        if df[column].dtype == object:
            df[column] = pd.to_numeric(df[column].str.replace('[\$,]', '', regex=True), errors='coerce')
        else:
            # If not an object type, attempt direct conversion to numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')
 
    return df

def clean_amenities(df, min_freq=0.1, max_freq=0.95, top_fallback=10):
    """
    Processes amenities data by:
    1. Cleaning raw amenities strings
    2. Creating dummy variables for common amenities
    3. Adding amenity count features
    
    Parameters:
    - df: Input DataFrame with 'amenities' column
    - min_freq: Minimum frequency threshold for amenity inclusion (0-1)
    - max_freq: Maximum frequency threshold for amenity inclusion (0-1)
    - top_fallback: Number of amenities to select if none meet frequency thresholds
    
    Returns:
    - DataFrame with cleaned amenities features
    """
    
    # Validate input
    if 'amenities' not in df.columns:
        raise ValueError("DataFrame must contain 'amenities' column")
        
    # Create working copy
    df_clean = df.copy()
    
    # Clean amenities
    def _clean_amenity_string(amenity_str):
        if pd.notna(amenity_str) and isinstance(amenity_str, str):
            cleaned = amenity_str.strip("{}")
            return [m[0] or m[1] for m in re.findall(r'"(.*?)"|([^,]+)', cleaned)]
        return []
    
    df_clean['cleaned_amenities'] = df_clean['amenities'].apply(_clean_amenity_string)
    
    # Analyze amenity frequency
    all_amenities = [a for sublist in df_clean['cleaned_amenities'] for a in sublist]
    amenity_counts = pd.Series(all_amenities).value_counts(normalize=True)
    
    # Select amenities
    selected = amenity_counts[(amenity_counts > min_freq) & (amenity_counts < max_freq)].index
    if len(selected) == 0:
        selected = amenity_counts.head(top_fallback).index
        print(f"⚠️ Using top {top_fallback} amenities as fallback")
    
    # Create dummy features
    dummy_data = pd.DataFrame({
        amenity: df_clean['cleaned_amenities'].apply(lambda x: 1 if amenity in x else 0)
        for amenity in selected
    })
    
    # Add features to DataFrame
    df_clean = pd.concat([df_clean, dummy_data.add_prefix('amenity_')], axis=1)
    df_clean['amenity_count'] = dummy_data.sum(axis=1)
    
    # Cleanup temporary column
    df_clean.drop(columns=['cleaned_amenities'], inplace=True)
    
    # Print diagnostics
    print(f"✅ Added {len(selected)} amenity features")
    print(f"Most common: {dummy_data.sum().idxmax()}")
    print(f"Least common: {dummy_data.sum().idxmin()}")
    print("amenitiy :::::::::::::::::::::::::::",df_clean)
    return df_clean

def clean_ccy_perct(df):
    """
    Cleans numerical columns containing currency values and percentages
    Handles: $ signs, commas, and percentage symbols
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Separate currency and percentage columns
    currency_columns = ["price", "security_deposit", "cleaning_fee", "extra_people"]
    percentage_columns = ["host_response_rate"]

    # Clean currency columns ($ and commas)
    for col in currency_columns:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace('[\$,]', '', regex=True)
                .replace('nan', '')  # Handle NaN string representations
                .replace('', np.nan)
                .astype(float)
            )
        else:
            print(f"⚠️ Warning: Column {col} not found in DataFrame")

    # Clean percentage columns
    for col in percentage_columns:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace('%', '', regex=False)
                .replace('nan', '')  # Handle NaN string representations
                .replace('', np.nan)
                .astype(float)
                / 100  # Convert percentage to decimal
            )
        else:
            print(f"⚠️ Warning: Column {col} not found in DataFrame")

    # Verification
    print("\n✅ Cleaned numerical columns:")
    print(df_clean[currency_columns + percentage_columns].dtypes)
    print("\n🔍 Sample values:")
    print(df_clean[currency_columns + percentage_columns].head())

    return df_clean

def fill_missing_values(df):
    """
    Fills missing values in the DataFrame with median or mode
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Fill missing values with Median for skewed variables
    median_fill_cols = [
    "host_response_rate", "host_listings_count",
    "security_deposit", "cleaning_fee", "reviews_per_month"
    ]

    df_clean.update(df_clean[median_fill_cols].fillna(df_clean[median_fill_cols].median()))

    # Fill missing values with Mode for categorical-like numerical variables
    mode_fill_cols = ["bathrooms", "bedrooms", "beds"]

    df_clean.update(df_clean[mode_fill_cols].apply(lambda x: x.fillna(x.mode()[0])))

    return df_clean

def winsorizing_outliers(df):
    """
    Winsorizes numerical columns to replace outliers with the nearest non-outlier"
    """
    # Define the lower and upper percentile thresholds for Winsorization (5th and 95th percentiles)
    lower_limit, upper_limit = np.percentile(df['price'], [5, 95])

    # Apply Winsorization directly to the 'price' column
    df['price'] = np.clip(df['price'], lower_limit, upper_limit)

    return df

def detect_unrealistic_listings(df):
    return df[
        (df["accommodates"] > 9) &  # High accommodates
        (
            (df["beds"] < df["accommodates"] / 2) |  # Too few beds for accommodates
            (df["bathrooms"] < df["accommodates"] / 4)  # Too few bathrooms for accommodates
        )
    ]

def clean_unrealistic(df):
    """
    Removes unrealistic listings based on the following criteria:
    - accommodates more than 9 guests
    - fewer beds than half the number of accommodates
    - fewer bathrooms than a quarter of the number of accommodates
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Detect unrealistic listings
    unrealistic_listings = detect_unrealistic_listings(df_clean)

    # **Remove unrealistic listings directly from df_listings**
    df_clean = df_clean.drop(unrealistic_listings.index).reset_index(drop=True)
    
    return df_clean

def check_skewness(df):
    numerical_columns = [
    "host_response_rate", "host_listings_count", "accommodates",
    "bathrooms", "bedrooms", "beds", "price", "security_deposit",
    "cleaning_fee", "guests_included", "extra_people", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms", "reviews_per_month", "review_scores_rating",
    ]
    # Compute skewness for each numerical variable
    skewness_values = df[numerical_columns].skew().sort_values(ascending=False)

    # Convert to DataFrame for easy interpretation
    skewness_df = pd.DataFrame(skewness_values, columns=["Skewness"]).reset_index()
    skewness_df.rename(columns={"index": "Column Name"}, inplace=True)

    return skewness_df

def clean_skewness(df):

    skewness_df = check_skewness(df)

    # Ensure skewness_df exists and contains the required columns
    if "Skewness" not in skewness_df.columns or "Column Name" not in skewness_df.columns:
        raise ValueError("Missing required columns in skewness_df. Ensure it has 'Skewness' and 'Column Name'.")

    # Identify highly skewed columns (skewness > 2)
    highly_skewed_cols = skewness_df[skewness_df['Skewness'] > 2]['Column Name'].tolist()

    # Filter out any columns that may not exist in df
    existing_skewed_cols = [col for col in highly_skewed_cols if col in df.columns]

    # Remove 'price' from the list of columns to be transformed
    existing_skewed_cols = [col for col in existing_skewed_cols if col != 'price']

    # Apply log transformation (log1p avoids log(0) errors)
    df[existing_skewed_cols] = df[existing_skewed_cols].apply(lambda x: np.log1p(x))

    return df

def drop_columns_numerical(df):
    columns_to_remove = ['availability_30', 'availability_90', 'log_number_of_reviews', 'log_host_listings_count']

    # Check if the columns exist before dropping
    columns_to_drop = [col for col in columns_to_remove if col in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Columns {columns_to_drop} removed successfully.")
    else:
        print("None of the specified columns were found in the DataFrame.")

    return df
 

#Aggregate function

def categorical_clean_data(df_clean):
    columns_to_encode = ['host_response_time', 'bed_type', 'room_type', 'property_type']
    date_columns = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']
    price_columns = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']

    #Start cleaning
    df_clean = clean_city(df_clean)
    df_clean = clean_street(df_clean)
    df_clean = clean_zipcode(df_clean)
    df_clean = clean_cancellation_policy(df_clean)
    df_clean = sklearn_ordinal_encoder(df_clean, columns_to_encode)
    df_clean = clean_date_columns(df_clean, date_columns)
    df_clean = clean_price_columns(df_clean, price_columns)  
    df_clean = clean_amenities(df_clean)
    bool_columns = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
    for col in bool_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.lower().map({'t': 1, 'f': 0})

    return df_clean

def numerical_clean_data(df_clean):
    df_clean = clean_ccy_perct(df_clean)
    df_clean = fill_missing_values(df_clean)
    df_clean = winsorizing_outliers(df_clean)
    df_clean = clean_unrealistic(df_clean)
    #Dropped numerical variables (threshold > 0.7 or < -0.7)
    df_clean = drop_columns_numerical(df_clean)

    return df_clean

def clean_data(df):
    df_clean = df.copy()
    df_clean = categorical_clean_data(df_clean)
    df_clean = numerical_clean_data(df_clean)

    return df_clean

df = pd.read_csv('listing.csv')
selected_features = [
    'bedrooms', 'room_type_encoded', 'amenity_Essentials', 'minimum_nights',
    'calculated_host_listings_count_entire_homes', 'cleaning_fee',
    'availability_60', 'host_is_superhost',
    'calculated_host_listings_count_shared_rooms', 'property_type_encoded',
    'calculated_host_listings_count_private_rooms', 'amenity_Shampoo',
    'host_response_time_encoded', 'amenity_Iron', 'amenity_Kitchen'
]
print(df.columns.to_list())
#  Start cleaning and running model

test = clean_data(df)
X = test[selected_features]

y = test['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')) 
                          if isinstance(x, str) else x)

 
valid_price_mask = y.between(20, 2000)   
X = X[valid_price_mask]
y = y[valid_price_mask]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

import joblib

# Save the trained model
joblib.dump(xgb_model, 'airbnb_price_predictor.pkl')