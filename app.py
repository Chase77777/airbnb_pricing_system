import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('airbnb_price_predictor.pkl')

# Define categorical mappings (replace with your actual encoding orders)
CATEGORY_MAPPINGS = {
    'room_type_encoded': {
        'Entire home/apt': 0,
        'Private room': 1,
        'Shared room': 2
    },
    'property_type_encoded': {
        'Apartment': 0,
        'House': 1,
        'Condominium': 2,
        'Guesthouse': 3,
        'Townhouse': 4
    },
    'host_response_time_encoded': {
        'within an hour': 0,
        'within a few hours': 1,
        'within a day': 2,
        'a few days or more': 3
    }
}

# Define order of features as per model training
FEATURE_ORDER = [
    'bedrooms', 'room_type_encoded', 'amenity_Essentials', 'minimum_nights',
    'calculated_host_listings_count_entire_homes', 'cleaning_fee',
    'availability_60', 'host_is_superhost',
    'calculated_host_listings_count_shared_rooms', 'property_type_encoded',
    'calculated_host_listings_count_private_rooms', 'amenity_Shampoo',
    'host_response_time_encoded', 'amenity_Iron', 'amenity_Kitchen'
]

def main():
    st.title('🏠 Airbnb Price Predictor')
    st.markdown("Predict optimal pricing for your Airbnb listing")

    with st.form("prediction_form"):
        # Numerical Inputs
        bedrooms = st.number_input('Number of Bedrooms', min_value=0, max_value=10, value=1)
        minimum_nights = st.number_input('Minimum Nights Stay', min_value=1, value=2)
        cleaning_fee = st.number_input('Cleaning Fee ($)', min_value=0.0, value=50.0)
        availability_60 = st.number_input('Availability (Next 60 Days)', min_value=0, max_value=60, value=30)
        
        # Categorical Inputs
        room_type = st.selectbox('Room Type', options=list(CATEGORY_MAPPINGS['room_type_encoded'].keys()))
        property_type = st.selectbox('Property Type', options=list(CATEGORY_MAPPINGS['property_type_encoded'].keys()))
        host_response_time = st.selectbox('Host Response Time', options=list(CATEGORY_MAPPINGS['host_response_time_encoded'].keys()))
        
        # Host Listings
        st.subheader("Host Listings Count")
        col1, col2, col3 = st.columns(3)
        with col1:
            entire_homes = st.number_input('Entire Homes', min_value=0, value=0)
        with col2:
            private_rooms = st.number_input('Private Rooms', min_value=0, value=0)
        with col3:
            shared_rooms = st.number_input('Shared Rooms', min_value=0, value=0)
        
        # Amenities
        st.subheader("Amenities")
        amen_col1, amen_col2, amen_col3 = st.columns(3)
        with amen_col1:
            essentials = st.checkbox('Essentials')
            shampoo = st.checkbox('Shampoo')
        with amen_col2:
            iron = st.checkbox('Iron')
            kitchen = st.checkbox('Kitchen')
        
        # Superhost
        superhost = st.radio("Are you a Superhost?", ('Yes', 'No'), horizontal=True)

        # Prediction button
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            # Create feature dictionary
            features = {
                'bedrooms': bedrooms,
                'room_type_encoded': CATEGORY_MAPPINGS['room_type_encoded'][room_type],
                'amenity_Essentials': int(essentials),
                'minimum_nights': minimum_nights,
                'calculated_host_listings_count_entire_homes': entire_homes,
                'cleaning_fee': cleaning_fee,
                'availability_60': availability_60,
                'host_is_superhost': 1 if superhost == 'Yes' else 0,
                'calculated_host_listings_count_shared_rooms': shared_rooms,
                'property_type_encoded': CATEGORY_MAPPINGS['property_type_encoded'][property_type],
                'calculated_host_listings_count_private_rooms': private_rooms,
                'amenity_Shampoo': int(shampoo),
                'host_response_time_encoded': CATEGORY_MAPPINGS['host_response_time_encoded'][host_response_time],
                'amenity_Iron': int(iron),
                'amenity_Kitchen': int(kitchen)
            }

            # Create DataFrame in correct order
            input_df = pd.DataFrame([features], columns=FEATURE_ORDER)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            st.success(f"**Recommended Price:** ${prediction:.2f} per night")
 

if __name__ == '__main__':
    main()