import streamlit as st
import pandas as pd
import io

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Assessment Scorer App (Two Headers)",
    layout="wide",
    initial_sidebar_state="expanded",
)

def process_multiindex_df(df):
    """
    Flattens a MultiIndex DataFrame's columns by concatenating the header rows.
    It cleans up and combines the section (Row 1) and question (Row 2).
    """
    # Create a list of new column names by combining level 0 and level 1
    new_columns = []
    for col_tuple in df.columns:
        # Convert to string and strip whitespace/NaN placeholders
        level0 = str(col_tuple[0]).strip()
        level1 = str(col_tuple[1]).strip()

        # Handle NaNs from Excel's merged cells or blank headers
        if 'Unnamed:' in level0:
            level0 = ''
        if 'Unnamed:' in level1:
            level1 = ''

        # If both levels are present, combine them.
        if level0 and level1:
            new_col_name = f"{level0} | {level1}"
        elif level0:
            new_col_name = level0
        else: # Should be caught by level1 if it's the only one present
            new_col_name = level1

        new_columns.append(new_col_name.replace('nan | ', '').replace(' | nan', ''))

    # Assign the new flat column names
    df.columns = new_columns
    
    # Drop rows that might have been part of the header but were included as data (usually the second header row)
    # This assumes the first non-NaN row in the 'Id' column is the start of the data.
    if 'Id' in df.columns:
        first_data_index = df['Id'].first_valid_index()
        if first_data_index is not None:
             df = df.loc[first_data_index:].reset_index(drop=True)

    return df

def identify_score_columns(df):
    """
    Identifies the columns to be scored based on keywords in the combined column names.
    The keywords are derived from the structure you provided.
    """
    # Keywords that indicate a column holds the PASS/FAIL evaluation
    score_keywords = [
        "Evaluate whether the prompt",
        "Evaluate the quality of the response",
    ]
    
    # Filter columns that contain any of the score keywords
    primary_score_columns = [
        col for col in df.columns 
        if any(keyword in col for keyword in score_keywords)
        # Exclude secondary explanation/rewrite columns that start with 'If your answer is'
        and not 'If your answer is' in col
        and not 'rewrite the prompt' in col.lower()
        and not 'rewrite the response' in col.lower()
    ]
             
    return sorted(primary_score_columns)


def calculate_score(df, score_cols):
    """
    Calculates the score for the identified columns.
    Score Logic: 'PASS' or 'Pass' = 1, 'FAIL' or 'Fail' = 0.
    """
    scored_df = df.copy()
    total_score_col = 'Total Assessment Score'
    scored_df[total_score_col] = 0
    
    for col in score_cols:
        # Convert to string and upper case to handle variations like 'pass', 'Pass', 'PASS'
        col_series = scored_df[col].astype(str).str.upper().str.strip()
        
        # Apply the scoring logic: 1 for PASS, 0 for everything else (including NaN/blanks/FAIL)
        # Note: We must check 'PASS' in x because the full content might be 'PASS (Minor Rewrite)'
        scored_df[f'SCORE: {col}'] = col_series.apply(lambda x: 1 if 'PASS' in x and not 'FAIL' in x else 0)
        
        # Add the score to the total
        scored_df[total_score_col] += scored_df[f'SCORE: {col}']
        
    return scored_df

def main():
    """Main function for the Streamlit App."""
    
    st.title("📊 Assessment Scoring App (Two-Row Header)")
    st.markdown("---")
    st.sidebar.header("Configuration")
    
    # --- 1. File Upload ---
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel/csv file (.xlsx or .xls or .csv)", 
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        try:
            # Read the Excel file, explicitly telling pandas that the header is in rows 0 and 1
            df_multiindex = pd.read_csv(uploaded_file, header=[0, 1])
            st.sidebar.success("File successfully uploaded with two header rows!")
            
            # --- Pre-processing: Flatten the headers ---
            with st.spinner("Processing two-row headers..."):
                df_flat = process_multiindex_df(df_multiindex)
            
            st.header("1. Data Preview (First 5 Rows - Flattened Headers)")
            st.info("The two header rows have been combined for easier scoring.")
            st.dataframe(df_flat.head())
            
            # --- 2. Column Identification ---
            score_cols = identify_score_columns(df_flat)
            
            st.header("2. Identified Scoring Columns")
            st.info(f"The app identified **{len(score_cols)}** primary columns for scoring:")
            
            # Show the first few columns
            col_list = "\n".join(score_cols[:5]) + ("\n..." if len(score_cols) > 5 else "")
            st.code(col_list)
            
            if not score_cols:
                st.error("Could not automatically identify scoring columns after flattening headers. Please check if the column names contain 'Evaluate whether the prompt' or 'Evaluate the quality of the response'.")
                return

            # --- 3. Score Calculation ---
            scored_df = calculate_score(df_flat, score_cols)

            st.header("3. Scoring Results")
            
            # --- 4. Display Results ---
            
            # Summary Statistics
            total_score_col = 'Total Assessment Score'
            max_score = len(score_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Scored Assessments (Rows)", value=len(scored_df))
            with col2:
                st.metric(label="Maximum Possible Score", value=max_score)
            
            st.subheader("Summary Score Statistics")
            st.dataframe(
                scored_df[total_score_col].describe().to_frame().T
            )

            # Full Scored DataFrame
            st.subheader("Full Scored Data")
            # Select key ID columns + the total score
            id_cols = ['Id', 'Full name (according to NRIC):'] 
            # Filter columns that are in the dataframe
            display_cols = [c for c in id_cols if c in scored_df.columns] + [total_score_col]
            # Add columns that start with 'SCORE: '
            display_cols.extend([col for col in scored_df.columns if col.startswith('SCORE: ')])
            
            st.dataframe(scored_df[display_cols])

            # Download Button
            @st.cache_data
            def convert_df_to_excel(df):
                """Converts the dataframe to an Excel file in memory."""
                output = io.BytesIO()
                # Use a specific sheet name
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Scored Assessments')
                processed_data = output.getvalue()
                return processed_data

            excel_data = convert_df_to_excel(scored_df)

            st.download_button(
                label="📥 Download Scored Data as Excel",
                data=excel_data,
                file_name='scored_assessments_processed.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.markdown("Please ensure your file is a valid Excel file and the first two rows contain the headers.")
            st.markdown("---")
            st.info("The column identification logic relies on the keywords 'Evaluate whether the prompt' and 'Evaluate the quality of the response' in the **combined** header.")

if __name__ == '__main__':
    main()