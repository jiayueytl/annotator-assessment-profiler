import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
import os

st.set_page_config(page_title="Annotator Assessment Profiler", layout="wide")

# --- CONFIGURATION ---
API_URL = "https://dot.ytlailabs.tech/api/v1/auth/token"
ANSWER_KEY_FILE = 'answer/answer_key.csv' 


# --- Utility Functions (Same) ---
def clean_answer(ans):
    if pd.isna(ans) or ans is None: return None
    ans_str = str(ans).upper().strip()
    if re.search(r'PASS', ans_str): return 'PASS'
    elif re.search(r'FAIL', ans_str): return 'FAIL'
    return None

def clean_answer(ans):
    if pd.isna(ans) or ans is None: return None
    ans_str = str(ans).upper().strip()
    if re.search(r'PASS', ans_str): return 'PASS'
    elif re.search(r'FAIL', ans_str): return 'FAIL'
    return None

def score_match(response_ans, key_ans):
    clean_response = clean_answer(response_ans)
    clean_key = clean_answer(key_ans)

    # NEW LOGIC: If both key and response are empty/null (None after cleaning), score 1.0
    if clean_key is None and clean_response is None: 
        return 1.0
        
    # If only one is None, we cannot score it (or it's a mismatch if you prefer 0.0)
    # Sticking to np.nan if only one is missing, as per the original intent (unscorable/NA).
    if clean_response is None or clean_key is None: 
        return np.nan 

    # Remaining scoring logic (PASS/FAIL comparison)
    if clean_key == clean_response: 
        return 1.0
    if clean_key == 'FAIL' and clean_response == 'PASS': 
        return 0.0
    elif clean_key == 'PASS' and clean_response == 'FAIL': 
        return 0.5
        
    return np.nan

def encode_confidence(value):
    if pd.isna(value): return np.nan
    if 'no' in str(value).lower(): return 0
    return 1

# --- Authentication (Same) ---
def login(username, password):
    payload = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "scope": "",
        "client_id": "string",
        "client_secret": "string"
    }
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        res = requests.post(API_URL, data=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            token = res.json().get("access_token")
            return token
        else:
            return None
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None


# --- Core Processing Logic (Updated to match previous functionality) ---
CONFIDENT_LANGUAGE_MAP = {
    'Unnamed: 15_level_0 | This question is about Bahasa Melayu (BM) language abilities.\nAre you confident reviewing and writing in both everyday/casual language (e.g. social media posts, chat messages) and more formal language': 'malay_confidence',
    'Unnamed: 16_level_0 | This question is about English language abilities.\nAre you confident reviewing and writing in both everyday/casual language (e.g. social media posts, chat messages) and more formal language (e.g. news': 'english_confidence',
    'Unnamed: 17_level_0 | This question is about Mandarin Chinese language abilities.\nAre you comfortable reviewing and writing in both everyday/casual language (e.g. social media posts, chat messages) and more formal language': 'chinese_confidence'
}
ID_COLUMNS_MAP = {
    'Unnamed: 6_level_0 | Full name (according to NRIC):': "name",
    'Unnamed: 9_level_0 | IC number:': "ic_number"
}
PROMPT_STATUS_PATTERN = "evaluate whether the prompt.*is usable.*"
RESPONSE_STATUS_PATTERN = "evaluate the quality of the response.*in the bolded blue text below.*"


def preprocess_responses(responses_raw):
    responses_raw.columns = [' | '.join(col).strip() if isinstance(col, tuple) else col for col in responses_raw.columns.values]
    responses = responses_raw.copy()
    column_map = {}

    for old_col in responses.columns:
        lower_col = old_col.lower().strip()
        parts = lower_col.split(' | ', 1)
        if len(parts) < 2: continue
        prefix, question = parts
        cleaned_prefix = re.sub(r'[^a-z0-9\s]', '', prefix).replace(' ', '_').strip('_')
        new_suffix = None
        if re.search(PROMPT_STATUS_PATTERN, question) or re.search(RESPONSE_STATUS_PATTERN, question):
            new_suffix = "status"
        if new_suffix:
            new_col_name = f"{cleaned_prefix}_{new_suffix}"
            column_map[old_col] = new_col_name

    responses.rename(columns=column_map, inplace=True)
    responses.rename(columns=CONFIDENT_LANGUAGE_MAP, inplace=True)
    
    for col in CONFIDENT_LANGUAGE_MAP.values():
        if col in responses.columns:
            responses[col] = responses[col].apply(encode_confidence)

    responses.rename(columns=ID_COLUMNS_MAP, inplace=True)
    
    scoring_columns = ['ic_number', 'name', 'malay_confidence', 'english_confidence', 'chinese_confidence'] + \
                      [col for col in responses.columns if col.startswith(("malay_", "english_", "chinese_"))]
    
    # Filter the DataFrame to only include relevant columns
    responses = responses.filter(items=scoring_columns)
    return responses

def generate_narrative_summary(responses_df_input, answer_key_df):
    responses_df = responses_df_input.copy()
    languages = ['malay', 'english', 'chinese']
    key_map = answer_key_df.set_index('question_key')['answer'].to_dict()
    
    def create_narrative(row, lang):
        lang_narrative_list = []
        q_score_cols = [col for col in row.index if col.startswith(f'{lang}_') and col.endswith('_score')]
        lang_total_score = row.get(f'{lang}_scores', 0.0)
        
        # Check Confidence
        confidence_col = f'{lang}_confidence'
        is_confident = row.get(confidence_col) == 1
        
        confidence_header = ""
        if not is_confident:
             # Add a warning note at the top of the narrative
             confidence_header = "⚠️ **Note:** Annotator marked **NOT CONFIDENT** in this language.\n\n"
        
        if not q_score_cols:
             return f"Total Score: **0.0** / 0 questions. No scorable questions for {lang}."

        for score_col in q_score_cols:
            # ... (rest of the question processing logic is unchanged)
            score = row.get(score_col)
            question_key = score_col.replace('_score', '')
            key_ans = key_map.get(question_key, 'KEY_NOT_FOUND')
            response_ans = row.get(question_key, 'RESPONSE_NOT_FOUND')
            
            # Determine status label
            if score == 1.0:
                status = "Match"
            elif score == 0.5:
                status = "Partial Mismatch (0.5)"
            elif score == 0.0:
                status = "Full Mismatch (0.0)"
            elif pd.isna(score):
                status = "Unscorable (NA)"
            else:
                status = "Unknown Score"

            # Create detailed explanation string (using triple quotes for multi-line format)
            explanation = f"""
Answer: '{key_ans}', 
Response: '{response_ans}'
"""
            
            # List item format: **Q_KEY** (Status: **Score**) -> Explanation
            lang_narrative_list.append(
                f"* **{question_key}** ({status} | Score: **{score:.1f}**): {explanation.strip()}"
            )

        mismatch_count = len([s for s in row[q_score_cols].values if s == 0.5 or s == 0.0])
        
        header = f"Total Score: **{lang_total_score:.1f}** / {len(q_score_cols)} questions."
        
        if mismatch_count > 0:
            header += f" ({mismatch_count} Mismatches)" 

        # Prepend the confidence warning
        return confidence_header + header + "\n\n" + "\n".join(lang_narrative_list)

    for lang in languages:
        responses_df[f'{lang}_narrative'] = responses_df.apply(
            lambda row: create_narrative(row, lang), axis=1
        )
        
    return responses_df[[f'{lang}_narrative' for lang in languages]]


def calculate_all_scores(responses_df_input, answer_key_df):
    responses_df = responses_df_input.copy()
    answer_key_df['clean_answer'] = answer_key_df['answer'].apply(clean_answer)
    
    # Calculate Individual Question Scores (Standard)
    for _, row in answer_key_df.iterrows():
        q_key = row['question_key']
        clean_key = row['clean_answer']
        if q_key in responses_df.columns:
            score_col_name = f"{q_key}_score"
            # Calculate standard score first (including 1.0 for nan/nan match)
            responses_df[score_col_name] = responses_df[q_key].apply(lambda resp_ans: score_match(resp_ans, clean_key))
            
    # Calculate Total Language Scores (Now implementing the 0.0 penalty for Not Confident)
    languages = ['malay', 'english', 'chinese']
    final_score_cols = []
    
    for lang in languages:
        prefix = f'{lang}_'
        confidence_col = f'{lang}_confidence'
        std_score_col_name = f'{prefix}scores'
        
        all_score_cols = [col for col in responses_df.columns if col.endswith('_score')]
        lang_score_cols = [col for col in all_score_cols if col.startswith(prefix)]

        # --- NEW LOGIC: Apply 0.0 penalty to individual question scores ---
        if confidence_col in responses_df.columns:
            is_not_confident = (responses_df[confidence_col] == 0)
            
            if is_not_confident.any():
                 # For all rows where the annotator is NOT CONFIDENT (0), 
                 # set their individual question scores for this language to 0.0
                 responses_df.loc[is_not_confident, lang_score_cols] = 0.0
        # ------------------------------------------------------------------
        
        if lang_score_cols:
            # Recalculate Total Score based on possibly penalized individual scores
            responses_df[std_score_col_name] = responses_df[lang_score_cols].sum(axis=1).round(1)
        else:
            responses_df[std_score_col_name] = 0.0
            
        final_score_cols.append(std_score_col_name)

    # Generate Narrative Summary
    # The narrative function will now use the 0.0 score from the penalized columns
    narrative_df = generate_narrative_summary(responses_df.copy(), answer_key_df) 
    
    # Final Output Construction
    final_output_cols = ['ic_number', 'name', 'malay_confidence', 'english_confidence', 'chinese_confidence'] + final_score_cols
    output_df = responses_df[final_output_cols].merge(
        narrative_df, left_index=True, right_index=True, how='left'
    )
    
    # Add a column for Total Confidence Count
    output_df['confident_count'] = output_df[['malay_confidence', 'english_confidence', 'chinese_confidence']].sum(axis=1).fillna(0).astype(int)

    # Rename for display
    output_df.rename(columns={
        'malay_scores': 'Malay Score (Total)',
        'english_scores': 'English Score (Total)',
        'chinese_scores': 'Chinese Score (Total)',
        'malay_narrative': 'Malay Breakdown',
        'english_narrative': 'English Breakdown',
        'chinese_narrative': 'Chinese Breakdown',
        'confident_count': 'Confident Lang Count'
    }, inplace=True)
    
    return output_df


# --- Streamlit Layout ---
st.title("🔐 Annotator Assessment Profiler Login")

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if not st.session_state.auth_token:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        token = login(username, password)
        if token:
            st.session_state.auth_token = token
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials or login failed.")
else:
    # --- LOGGED IN VIEW ---
    st.success("✅ Logged in successfully")
    st.divider()

    # 1. Load Answer Key (from fixed directory)
    if not os.path.exists(ANSWER_KEY_FILE):
        st.error(f"❌ **Error:** Answer Key file not found. Please ensure `{ANSWER_KEY_FILE}` is accessible.")
        # Offer a temporary fallback if needed, but per instruction, it's fixed.
        if st.button("Logout"):
            st.session_state.auth_token = None
            st.rerun()
        st.stop()
    
    try:
        # NOTE: Using header=None and names due to previous multi-column structure
        answer_key_df = pd.read_csv(
            ANSWER_KEY_FILE, 
            sep='\t', 
            header=None, 
            skiprows=1,
            names=['language', 'question_num', 'answer_type', 'answer', 'question_key']
        )
        st.info(f"Answer Key successfully loaded from `{ANSWER_KEY_FILE}` (Questions: {len(answer_key_df)}).")
    except Exception as e:
        st.error(f"❌ **Error** loading Answer Key from `{ANSWER_KEY_FILE}`. Please check file format.")
        st.exception(e)
        st.stop()


    # 2. File Uploader for Responses
    st.header("📂 Upload Raw Responses")
    response_file = st.file_uploader(
        "Upload Raw Assessment Responses (.csv)", 
        type=["csv"],
        help="This file is expected to have a two-row header."
    )

    if response_file:
        with st.spinner("Processing data..."):
            try:
                responses_raw = pd.read_csv(response_file, header=[0, 1])
                processed = preprocess_responses(responses_raw)
                final_output = calculate_all_scores(processed, answer_key_df)

                st.success("✅ Processing complete and results generated!")

                SCORE_COLS = ['Malay Score (Total)', 'English Score (Total)', 'Chinese Score (Total)']
                
                # --- PIVOT TABLE (Summary) ---
                st.header("📊 Scoring Summary by Name")
                pivot_df = final_output.pivot_table(
                    index=['name', 'ic_number'],
                    values=SCORE_COLS,
                    aggfunc='mean'
                ).sort_values(by=SCORE_COLS[0], ascending=False)
                
                st.dataframe(pivot_df.style.background_gradient(cmap='YlGnBu', axis=None).format('{:.1f}'), use_container_width=True)

                # --- CONFIDENCE-BASED SCORING ---
                st.header("🧠 Analysis by Language Confidence")
                
                confidence_groups = {
                    3: '🌟 All 3 Languages (Malay, English, Chinese)',
                    2: '📘 2 Languages Only',
                    1: '🔸 1 Language Only',
                    0: '❓ None of the Languages'
                }

                for count, label in confidence_groups.items():
                    group_df = final_output[final_output['Confident Lang Count'] == count]
                    if not group_df.empty:
                        with st.expander(f"{label} (N={len(group_df)})", expanded=True if count == 3 else False):
                            group_pivot = group_df.pivot_table(
                                index='name',
                                values=SCORE_COLS,
                                aggfunc='mean'
                            ).sort_values(by=SCORE_COLS[0], ascending=False)
                            st.dataframe(group_pivot.style.background_gradient(cmap='YlGn').format('{:.1f}'), use_container_width=True)
                            
                
                # --- DETAILED RESULTS & BREAKDOWN ---
                st.header("📋 Detailed Per-Respondent Results & Breakdown")
                st.markdown("Click on the **+** for the detailed score narrative.")
                
                for index, row in final_output.iterrows():
                    with st.expander(f"👤 **{row['name']}** ({row['ic_number']}) | Total Score: **{row[SCORE_COLS].sum():.1f}**"):
                        
                        st.markdown(f"**Confident in:** {row['Confident Lang Count']} of 3 languages")

                        # 1 Row, 3 Columns for Breakdowns
                        col_m, col_e, col_c = st.columns(3)
                        
                        with col_m:
                            st.markdown("### 🇲🇾 Malay Score")
                            st.markdown(row['Malay Breakdown'])
                            
                        with col_e:
                            st.markdown("### 🇬🇧 English Score")
                            st.markdown(row['English Breakdown'])
                            
                        with col_c:
                            st.markdown("### 🇨🇳 Chinese Score")
                            st.markdown(row['Chinese Breakdown'])
                            

                # Download button
                csv_buffer = BytesIO()
                final_output.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Full Results CSV",
                    data=csv_buffer.getvalue(),
                    file_name="annotator_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ An error occurred during processing. Please check your file formats and column structure.")
                st.exception(e)
            
    st.divider()
    if st.button("🔒 Logout"):
        st.session_state.auth_token = None
        st.rerun()