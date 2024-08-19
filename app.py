import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

# Function to load and display data head
def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding='ISO-8859-1')  # Try 'ISO-8859-1' or 'latin1'
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
            
            st.subheader('Preview of Data')
            st.write(df.head())
            st.write(f'**Data Shape**: {df.shape}')

            # Display data description
            st.subheader('Data Description')
            st.write(df.describe(include='all'))  # include='all' provides description for all types of columns

            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None


# Function to parse and handle dates
def parse_date(date_str):
    if isinstance(date_str, str):
        parts = date_str.split('/')
        
        # Ensure the date has exactly three parts
        if len(parts) == 3:
            day, month, year = parts
            
            # Check for invalid day, month, or year
            if day == '00' or month == '00' or len(year) != 4 or not year.isdigit():
                return pd.NaT
            
            # Construct a valid date string if parts are not '00'
            try:
                return pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
            except ValueError:
                return pd.NaT
        else:
            return pd.NaT
    else:
        return pd.NaT

# Function to convert columns to datetime
def convert_column_to_datetime(df, columns):
    if df is not None:
        for column in columns:
            if column in df.columns:
                st.write(f"Before conversion - Column: {column}")
                st.write(df[column].head(10))  # Display the first 10 values for debugging
                df[column] = df[column].apply(parse_date)
                st.write(f"After conversion - Column: {column}")
                st.write(df[column].head(10))  # Display the first 10 values after conversion for debugging
        return df
    return None

# Function to remove duplicates
def remove_duplicates(df):
    if df is not None:
        original_len = len(df)
        df_cleaned = df.drop_duplicates()
        removed_len = original_len - len(df_cleaned)
        st.write(f"Removed {removed_len} duplicate rows.")
        st.write(f"Data Shape after removing duplicates: {df_cleaned.shape}")
        return df_cleaned
    return None

# Function to convert DataFrame to CSV and generate download link
def get_csv_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Cleaned Data (CSV)</a>'
    return href

# Function to handle missing values by filling with mean
def fill_missing_with_mean(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_mean = df.copy()
        for col in columns:
            if df_filled_mean[col].dtype in ['float64', 'int64'] and df_filled_mean[col].isnull().sum() > 0:  # Only fill if there are missing values
                mean_value = df_filled_mean[col].mean()
                df_filled_mean[col].fillna(mean_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mean: {mean_value}")
        st.write("After filling with mean:")
        st.write(df_filled_mean.head())
        return df_filled_mean
    return None

# Function to handle missing values by filling with median
def fill_missing_with_median(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_median = df.copy()
        for col in columns:
            if df_filled_median[col].dtype in ['float64', 'int64'] and df_filled_median[col].isnull().sum() > 0:  # Only fill if there are missing values
                median_value = df_filled_median[col].median()
                df_filled_median[col].fillna(median_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with median: {median_value}")
        st.write("After filling with median:")
        st.write(df_filled_median.head())
        return df_filled_median
    return None

# Function to handle missing values by filling with mode for object data types
def fill_missing_with_mode(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_mode = df.copy()
        for col in columns:
            if df_filled_mode[col].dtype == 'object' and df_filled_mode[col].isnull().sum() > 0:  # Only fill if there are missing values
                mode_value = df_filled_mode[col].mode()[0]
                df_filled_mode[col].fillna(mode_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mode: {mode_value}")
        st.write("After filling with mode:")
        st.write(df_filled_mode.head())
        return df_filled_mode
    return None

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_no_outliers = df.copy()
        any_outliers_removed = False
        
        for col in columns:
            if df_no_outliers[col].dtype in ['float64', 'int64']:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                before_shape = df_no_outliers.shape
                df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
                after_shape = df_no_outliers.shape
                if before_shape != after_shape:
                    any_outliers_removed = True
                    st.write(f"Removed outliers in column '{col}' using IQR. Rows before: {before_shape[0]}, Rows after: {after_shape[0]}")
        
        if not any_outliers_removed:
            st.write("No outliers found in the selected columns.")
        else:
            st.write("After removing outliers:")
            st.write(df_no_outliers.head())
        
        return df_no_outliers
    return None

# Function to remove numeric values from a predominantly categorical column
def remove_numeric_from_categorical(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(lambda x: x if not str(x).isdigit() else np.nan)
    return df

# Main function
def main():
    st.title('Automated Pre-processing')
    st.markdown("""
    ## Automated Pre-processing of Data
    Upload your dataset and select columns to remove, handle missing values, duplicates, and outliers.
    """)

    # Initialize session state variables if they do not exist
    if 'df_datetime_converted' not in st.session_state:
        st.session_state.df_datetime_converted = None
    if 'df_deduplicated' not in st.session_state:
        st.session_state.df_deduplicated = None
    if 'df_mean_filled' not in st.session_state:
        st.session_state.df_mean_filled = None
    if 'df_median_filled' not in st.session_state:
        st.session_state.df_median_filled = None
    if 'df_mode_filled' not in st.session_state:
        st.session_state.df_mode_filled = None

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        if df is not None:
            # Store the original DataFrame in session state
            st.session_state.df_original = df

            # Convert columns to datetime
            st.subheader('Convert Columns to Datetime')
            datetime_columns_key = 'datetime_columns_select'
            
            selected_datetime_columns = st.multiselect(
                "Select columns to convert to datetime",
                df.columns,
                key=datetime_columns_key
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Convert to Datetime"):
                    if selected_datetime_columns:
                        st.session_state.df_datetime_converted = convert_column_to_datetime(df, selected_datetime_columns)
                        st.write("After converting columns to datetime:")
                        st.write(st.session_state.df_datetime_converted.head())
                    else:
                        st.warning("Please select at least one column to convert to datetime.")
                
                if st.button("Skip Datetime Conversion"):
                    st.session_state.df_datetime_converted = df

            # Convert non-numeric values to NaN
            st.subheader('Convert Non-Numeric Values to NaN')
            if st.session_state.df_datetime_converted is not None and not st.session_state.df_datetime_converted.empty:
                object_columns = st.session_state.df_datetime_converted.select_dtypes(include='object').columns.tolist()
                non_numeric_columns_key = 'non_numeric_columns_select'
                
                selected_non_numeric_columns = st.multiselect(
                    "Select columns to handle non-numeric values",
                    object_columns,
                    key=non_numeric_columns_key
                )
                
                if st.button("Handle Non-Numeric Values"):
                    df_no_non_numeric = st.session_state.df_datetime_converted.copy()
                    for col in selected_non_numeric_columns:
                        df_no_non_numeric[col] = pd.to_numeric(df_no_non_numeric[col], errors='coerce')
                    st.session_state.df_datetime_converted = df_no_non_numeric
                    st.write("After handling non-numeric values:")
                    st.write(st.session_state.df_datetime_converted.head())
                
                if st.button("Skip Non-Numeric Value Handling"):
                    st.session_state.df_datetime_converted = st.session_state.df_datetime_converted

            # Convert numeric values in categorical columns to NaN
            st.subheader('Convert Numeric Values in Categorical Columns to NaN')
            if st.session_state.df_datetime_converted is not None and not st.session_state.df_datetime_converted.empty:
                object_columns_with_numeric = st.session_state.df_datetime_converted.select_dtypes(include='object').columns.tolist()
                numeric_columns_key = 'numeric_columns_select'
                
                selected_numeric_columns = st.multiselect(
                    "Select categorical columns with numeric values to handle",
                    object_columns_with_numeric,
                    key=numeric_columns_key
                )
                
                if st.button("Handle Numeric Values in Categorical Columns"):
                    df_no_numeric = remove_numeric_from_categorical(st.session_state.df_datetime_converted, selected_numeric_columns[0])
                    st.session_state.df_datetime_converted = df_no_numeric
                    st.write("After handling numeric values in categorical columns:")
                    st.write(st.session_state.df_datetime_converted.head())
                
                if st.button("Skip Numeric Value Handling"):
                    st.session_state.df_datetime_converted = st.session_state.df_datetime_converted

            # Remove duplicates
            st.subheader('Remove Duplicates')
            if st.session_state.df_datetime_converted is not None and not st.session_state.df_datetime_converted.empty:
                if st.button("Remove Duplicates"):
                    st.session_state.df_deduplicated = remove_duplicates(st.session_state.df_datetime_converted)
                    st.write(st.session_state.df_deduplicated.head())

                    # Display data info directly
                    st.subheader('Data Information')
                    buffer = io.StringIO()
                    st.session_state.df_deduplicated.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                if st.button("Skip Duplicate Removal"):
                    st.session_state.df_deduplicated = st.session_state.df_datetime_converted

            # Fill missing values with mean
            st.subheader('Fill Missing Values with Mean')
            if st.session_state.df_deduplicated is not None and not st.session_state.df_deduplicated.empty:
                numeric_columns = st.session_state.df_deduplicated.select_dtypes(include=['float64', 'int64']).columns.tolist()
                mean_fill_columns_key = 'mean_fill_columns_select'
                
                selected_mean_fill_columns = st.multiselect(
                    "Select numeric columns to fill missing values with mean",
                    numeric_columns,
                    key=mean_fill_columns_key
                )
                
                if st.button("Fill Missing Values with Mean"):
                    st.session_state.df_mean_filled = fill_missing_with_mean(st.session_state.df_deduplicated, selected_mean_fill_columns)
                    st.write("After filling with mean:")
                    st.write(st.session_state.df_mean_filled.head())
                
                if st.button("Skip Mean Filling"):
                    st.session_state.df_mean_filled = st.session_state.df_deduplicated

            # Fill missing values with median
            st.subheader('Fill Missing Values with Median')
            if st.session_state.df_mean_filled is not None and not st.session_state.df_mean_filled.empty:
                numeric_columns = st.session_state.df_mean_filled.select_dtypes(include=['float64', 'int64']).columns.tolist()
                median_fill_columns_key = 'median_fill_columns_select'
                
                selected_median_fill_columns = st.multiselect(
                    "Select numeric columns to fill missing values with median",
                    numeric_columns,
                    key=median_fill_columns_key
                )
                
                if st.button("Fill Missing Values with Median"):
                    st.session_state.df_median_filled = fill_missing_with_median(st.session_state.df_mean_filled, selected_median_fill_columns)
                    st.write("After filling with median:")
                    st.write(st.session_state.df_median_filled.head())
                
                if st.button("Skip Median Filling"):
                    st.session_state.df_median_filled = st.session_state.df_mean_filled

            # Fill missing values with mode
            st.subheader('Fill Missing Values with Mode')
            if st.session_state.df_median_filled is not None and not st.session_state.df_median_filled.empty:
                object_columns = st.session_state.df_median_filled.select_dtypes(include='object').columns.tolist()
                mode_fill_columns_key = 'mode_fill_columns_select'
                
                selected_mode_fill_columns = st.multiselect(
                    "Select object columns to fill missing values with mode",
                    object_columns,
                    key=mode_fill_columns_key
                )
                
                if st.button("Fill Missing Values with Mode"):
                    st.session_state.df_mode_filled = fill_missing_with_mode(st.session_state.df_median_filled, selected_mode_fill_columns)
                    st.write("After filling with mode:")
                    st.write(st.session_state.df_mode_filled.head())

                    # Provide download link for the DataFrame without mode filling
                    st.markdown(get_csv_download_link(st.session_state.df_mode_filled, filename="cleaned_data_mode_filled.csv"), unsafe_allow_html=True)
                
                if st.button("Skip Mode Filling"):
                    st.session_state.df_mode_filled = st.session_state.df_median_filled
                    st.write(st.session_state.df_mode_filled.head())
                    
                    # Provide download link for the DataFrame without mode filling
                    st.markdown(get_csv_download_link(st.session_state.df_mode_filled, filename="cleaned_data_mode_filled.csv"), unsafe_allow_html=True)
                    
            # Remove outliers
            st.subheader('Remove Outliers')
            if st.session_state.df_mode_filled is not None and not st.session_state.df_mode_filled.empty:
                numeric_columns = st.session_state.df_mode_filled.select_dtypes(include=['float64', 'int64']).columns.tolist()
                outlier_columns_key = 'outlier_columns_select'
                
                selected_outlier_columns = st.multiselect(
                    "Select numeric columns to remove outliers",
                    numeric_columns,
                    key=outlier_columns_key
                )
                
                if st.button("Remove Outliers"):
                    df_no_outliers = remove_outliers_iqr(st.session_state.df_mode_filled, selected_outlier_columns)
                    st.write("After removing outliers:")
                    st.write(df_no_outliers.head())
                    
                    # Provide download link for the DataFrame after removing outliers
                    st.markdown(get_csv_download_link(df_no_outliers, filename="cleaned_data_no_outliers.csv"), unsafe_allow_html=True)
                
                if st.button("Skip Outlier Removal"):
                    st.write(st.session_state.df_mode_filled.head())

# Run the Streamlit app
if __name__ == "__main__":
    main()
