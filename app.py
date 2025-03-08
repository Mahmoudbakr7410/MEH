from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import logging
import math
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Initialize session state variables
app.config['df'] = None
app.config['processed_df'] = None
app.config['public_holidays'] = []
app.config['high_risk_entries'] = None
app.config['rounded_threshold'] = 100
app.config['column_mapping'] = {}
app.config['authorized_users'] = []
app.config['closing_date'] = None
app.config['logged_in'] = False
app.config['auth_threshold'] = 10000
app.config['suspicious_keywords'] = []
app.config['trial_balance'] = None
app.config['completeness_check_results'] = None
app.config['completeness_check_passed'] = False
app.config['audited_client_name'] = ""
app.config['year_audited'] = datetime.now().year
app.config['flagged_entries_by_category'] = {}
app.config['pattern_recognition_results'] = None
app.config['seldomly_used_accounts_threshold'] = 5

# Define authorized users
authorized_users = {
    "a.habbul@maham.com": "password1",
    "a.elnahal@maham.com": "password2",
    "a.younes@maham.com": "password3",
    "a.alhazmi@maham.com": "password4",
    "a.almousa@maham.com": "password5",
    "a.alqadi@maham.com": "password6",
    "a.alqahtani@maham.com": "password7",
    "a.alrubayan@maham.com": "password8",
    "a.alamodi@maham.com": "password9",
    "a.alremawi@maham.com": "password10",
    "a.abdelgawad@maham.com": "password11",
    "a.alwhaibi@maham.com": "password12",
    "a.elnouby@maham.com": "password13",
    "a.goma@maham.com": "password14",
    "a.magdi@maham.com": "password15",
    "a.nagy@maham.com": "password16",
    "a.basith@maham.com": "password17",
    "a.alali@maham.com": "password18",
    "a.arafat@maham.com": "password19",
    "a.shedeed@maham.com": "password20",
    "a.salem@maham.com": "password21",
    "a.khan@maham.com": "password22",
    "E.Alshehri@maham.com": "password23",
    "f.alkaltham@maham.com": "password24",
    "f.alanazi@maham.com": "password25",
    "f.muhammad@maham.com": "password26",
    "I.abdulwahab@maham.com": "password27",
    "i.alabdullah@maham.com": "password28",
    "i.alotaibi@maham.com": "password29",
    "i.metwally@maham.com": "password30",
    "j.rizkallah@maham.com": "password31",
    "kh.almatroudi@maham.com": "password32",
    "l.Alrizqi@maham.com": "password33",
    "l.altuwaim@maham.com": "password34",
    "m.abead@maham.com": "password35",
    "m.abdelrahim@maham.com": "password36",
    "m.elansary@maham.com": "74107410",
    "m.hamouda@maham.com": "password37",
    "m.mostafa@maham.com": "password38",
    "m.noman@maham.com": "password39",
    "m.erman@maham.com": "password40",
    "m.alqattan@maham.com": "password41",
    "m.alrashidi@maham.com": "password42",
    "M.Alshammari@maham.com": "password43",
    "m.bilal@maham.com": "password44",
    "m.zain@maham.com": "password45",
    "m.alangari@maham.com": "password46",
    "m.attia@maham.com": "password47",
    "m.Thafseem@maham.com": "password48",
    "m.masood@maham.com": "password49",
    "m.Alshehri@maham.com": "password50",
    "n.adham@maham.com": "password51",
    "n.alsayeh@maham.com": "password52",
    "n.sabhah@maham.com": "password53",
    "o.almatrudi@maham.com": "password54",
    "r.alabdulhadi@maham.com": "password55",
    "r.alhamidi@maham.com": "password56",
    "r.aljebali@maham.com": "password57",
    "s.uddin@maham.com": "password58",
    "s.alharbi@maham.com": "password59",
    "s.salih@maham.com": "password60",
    "s.ahmed@maham.com": "password61",
    "s.alqadi@maham.com": "password62",
    "s.lashaera@maham.com": "password63",
    "sh.alanazi@maham.com": "password64",
    "s.alhalal@maham.com": "password65",
    "t.alhassan@maham.com": "password66",
    "u.riaz@maham.com": "password67",
    "w.alanazi@maham.com": "password68",
    "s.habib@maham.com": "internalaudit@2025",
    "y.alahmadi@maham.com": "password69"
}

# Define required and optional fields
required_fields = [
    "Transaction ID", "Date", "Debit Amount (Dr)", "Credit Amount (Cr)", "Account Number"
]

optional_fields = [
    "Journal Entry ID", "Posting Date", "Entry Description", "Document Number",
    "Period/Month", "Year", "Entry Type", "Reversal Indicator", "Account Name",
    "Account Type", "Cost Center", "Subledger Type", "Subledger ID", "Currency", "Local Currency Amount",
    "Exchange Rate", "Net Amount", "Created By", "Approved By", "Posting User", "Approval Date",
    "Journal Source", "Manual Entry Flag", "High-Risk Account Flag", "Suspense Account Flag",
    "Offsetting Entry Indicator", "Period-End Flag", "Weekend/Holiday Flag", "Round Number Flag"
]

all_fields = required_fields + optional_fields

# Function to detect delimiter for txt files
def detect_delimiter(file):
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    return delimiter

# Function to convert data types
def convert_data_types(df):
    numeric_fields = ["Debit Amount (Dr)", "Credit Amount (Cr)"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    date_fields = ["Date"]
    for field in date_fields:
        if field in df.columns:
            df[field] = pd.to_datetime(df[field], errors="coerce")
    return df

# Function to check for 99999 pattern
def is_99999(value):
    try:
        value = float(value)
        return abs(value - round(value, 0)) >= 0.999 and abs(value - round(value, 0)) < 1.0
    except (ValueError, TypeError):
        return False

# Function to perform completeness check
def perform_completeness_check():
    if app.config['processed_df'] is None or app.config['processed_df'].empty:
        return "No GL data to test. Please import a file first."
    if app.config['trial_balance'] is None or app.config['trial_balance'].empty:
        return "No trial balance data to test. Please import a trial balance file first."

    try:
        gl_summary = app.config['processed_df'].groupby("Account Number").agg(
            Total_Debits=("Debit Amount (Dr)", "sum"),
            Total_Credits=("Credit Amount (Cr)", "sum")
        ).reset_index()
        merged_df = pd.merge(
            app.config['trial_balance'],
            gl_summary,
            on="Account Number",
            how="left"
        )
        merged_df["Total_Debits"] = merged_df["Total_Debits"].fillna(0)
        merged_df["Total_Credits"] = merged_df["Total_Credits"].fillna(0)
        merged_df["Expected_Ending_Balance"] = (
            merged_df["Opening Balance"] + merged_df["Total_Debits"] - merged_df["Total_Credits"]
        )
        merged_df["Discrepancy"] = (
            merged_df["Expected_Ending_Balance"] - merged_df["Ending Balance"]
        )
        app.config['completeness_check_results'] = merged_df
        max_discrepancy = merged_df["Discrepancy"].abs().max()
        if max_discrepancy <= 5:
            app.config['completeness_check_passed'] = True
            return "Completeness check passed! Maximum discrepancy is within the allowed tolerance of 5."
        else:
            app.config['completeness_check_passed'] = False
            return f"Completeness check failed! Maximum discrepancy ({max_discrepancy}) exceeds the allowed tolerance of 5."
    except Exception as e:
        return f"Error during completeness check: {e}"

# Function to detect seldomly used accounts
def detect_seldomly_used_accounts():
    if app.config['processed_df'] is None or app.config['processed_df'].empty:
        return "No data to analyze. Please import a file first."

    try:
        account_frequency = app.config['processed_df']["Account Number"].value_counts().reset_index()
        account_frequency.columns = ["Account Number", "Transaction Count"]
        seldomly_used_accounts = account_frequency[account_frequency["Transaction Count"] < app.config['seldomly_used_accounts_threshold']]
        app.config['seldomly_used_accounts'] = seldomly_used_accounts
        return f"Found {len(seldomly_used_accounts)} accounts with fewer than {app.config['seldomly_used_accounts_threshold']} transactions."
    except Exception as e:
        return f"Error during seldomly used accounts detection: {e}"

# Function to perform data mining and pattern recognition
def perform_pattern_recognition():
    if app.config['processed_df'] is None or app.config['processed_df'].empty:
        return "No data to analyze. Please import a file first."

    try:
        numeric_cols = app.config['processed_df'].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric columns found for pattern recognition."
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(app.config['processed_df'][numeric_cols])
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(scaled_data)
        app.config['processed_df']["Cluster"] = clusters
        cluster_summary = app.config['processed_df'].groupby("Cluster").agg(
            Count=("Cluster", "size"),
            Avg_Debit=("Debit Amount (Dr)", "mean"),
            Avg_Credit=("Credit Amount (Cr)", "mean")
        ).reset_index()
        app.config['pattern_recognition_results'] = cluster_summary
        return "Pattern recognition identified distinct groups of transactions. Review the clusters for insights."
    except Exception as e:
        return f"Error during pattern recognition: {e}"

# Function to export PDF report
def export_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Maham for Professional Services", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Audited Client: {app.config['audited_client_name']}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Year Audited: {app.config['year_audited']}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Generated By: {app.config['logged_in_user']}", ln=True, align="L")
    pdf.cell(200, 10, txt="Completeness Check Conclusion:", ln=True, align="L")
    if app.config['completeness_check_passed']:
        pdf.cell(200, 10, txt="Completeness check passed. Maximum discrepancy is within the allowed tolerance of 5.", ln=True, align="L")
    else:
        max_discrepancy = app.config['completeness_check_results']["Discrepancy"].abs().max()
        pdf.cell(200, 10, txt=f"Completeness check failed. Maximum discrepancy ({max_discrepancy}) exceeds the allowed tolerance of 5.", ln=True, align="L")
    pdf.cell(200, 10, txt="Flagged Entries by Category:", ln=True, align="L")
    pdf.set_font("Arial", size=10)
    for category, entries in app.config['flagged_entries_by_category'].items():
        pdf.cell(200, 10, txt=f"Category: {category}", ln=True, align="L")
        for index, row in entries.iterrows():
            pdf.cell(200, 10, txt=f"Transaction ID: {row['Transaction ID']}, Date: {row['Date']}, Debit: {row['Debit Amount (Dr)']}, Credit: {row['Credit Amount (Cr)']}", ln=True, align="L")
    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output

# Function to export Excel report
def export_excel_report():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for category, entries in app.config['flagged_entries_by_category'].items():
            entries.to_excel(writer, sheet_name=category, index=False)
    output.seek(0)
    return output

# Function to perform high-risk testing
def perform_high_risk_test():
    if not app.config['completeness_check_passed']:
        return "High-risk tests are disabled until the completeness check passes with a maximum discrepancy of 5."

    if app.config['processed_df'] is None or app.config['processed_df'].empty:
        return "No data to test. Please import a file first."

    try:
        app.config['high_risk_entries'] = pd.DataFrame()
        app.config['flagged_entries_by_category'] = {}

        if app.config['public_holidays_var']:
            if "Date" in app.config['processed_df'].columns:
                holiday_entries = app.config['processed_df'][app.config['processed_df']["Date"].isin(app.config['public_holidays'])]
                app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], holiday_entries])
                app.config['flagged_entries_by_category']["Public Holidays"] = holiday_entries
            else:
                return "Column 'Date' not found in the data."

        if app.config['rounded_var']:
            def is_rounded(value, threshold):
                try:
                    value = float(value)
                    if value == 0:
                        return False
                    return (value % threshold == 0) or (math.isclose(value % threshold, threshold, rel_tol=1e-6))
                except (ValueError, TypeError):
                    return False

            rounded_entries = app.config['processed_df'][
                app.config['processed_df']["Debit Amount (Dr)"].apply(lambda x: is_rounded(x, app.config['rounded_threshold'])) |
                app.config['processed_df']["Credit Amount (Cr)"].apply(lambda x: is_rounded(x, app.config['rounded_threshold']))
            ]
            app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], rounded_entries])
            app.config['flagged_entries_by_category']["Rounded Numbers"] = rounded_entries

        if app.config['unusual_users_var']:
            if "Created By" in app.config['processed_df'].columns:
                if not app.config['authorized_users']:
                    return "No authorized users provided. Skipping unusual users check."
                else:
                    unusual_user_entries = app.config['processed_df'][~app.config['processed_df']["Created By"].isin(app.config['authorized_users'])]
                    app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], unusual_user_entries])
                    app.config['flagged_entries_by_category']["Unauthorized Users"] = unusual_user_entries
            else:
                return "Column 'Created By' not found in the data."

        if app.config['post_closing_var']:
            if "Date" in app.config['processed_df'].columns:
                if app.config['closing_date'] is None:
                    return "No closing date provided. Skipping post-closing entries check."
                else:
                    # Convert closing_date to datetime64[ns]
                    closing_date = pd.to_datetime(app.config['closing_date'])
                    # Ensure the Date column is datetime64[ns]
                    app.config['processed_df']["Date"] = pd.to_datetime(app.config['processed_df']["Date"])
                    # Restrict post_closing_date to be after the audited year's December 31
                    audited_year_end = pd.to_datetime(f"{app.config['year_audited']}-12-31")
                    if closing_date <= audited_year_end:
                        return "Closing date must be after the audited year's December 31."
                    post_closing_entries = app.config['processed_df'][app.config['processed_df']["Date"] > closing_date]
                    app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], post_closing_entries])
                    app.config['flagged_entries_by_category']["Post-Closing Entries"] = post_closing_entries
            else:
                return "Column 'Date' not found in the data."

        if app.config['auth_threshold_var']:
            threshold = app.config['auth_threshold']
            below_threshold_entries = app.config['processed_df'][
                (app.config['processed_df']["Debit Amount (Dr)"] >= threshold * 0.9) & 
                (app.config['processed_df']["Debit Amount (Dr)"] < threshold) |
                (app.config['processed_df']["Credit Amount (Cr)"] >= threshold * 0.9) & 
                (app.config['processed_df']["Credit Amount (Cr)"] < threshold)
            ]
            app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], below_threshold_entries])
            app.config['flagged_entries_by_category']["Below Authorization Threshold"] = below_threshold_entries

        if app.config['nine_pattern_var']:
            nine_pattern_entries = app.config['processed_df'][
                app.config['processed_df']["Debit Amount (Dr)"].apply(is_99999) |
                app.config['processed_df']["Credit Amount (Cr)"].apply(is_99999)
            ]
            app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], nine_pattern_entries])
            app.config['flagged_entries_by_category']["99999 Pattern"] = nine_pattern_entries

        if app.config['keywords_var']:
            if "Entry Description" in app.config['processed_df'].columns:
                if not app.config['suspicious_keywords']:
                    return "No suspicious keywords provided. Skipping keyword check."
                else:
                    keyword_entries = app.config['processed_df'][
                        app.config['processed_df']["Entry Description"].str.contains(
                            "|".join(app.config['suspicious_keywords']), case=False, na=False
                        )
                    ]
                    app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], keyword_entries])
                    app.config['flagged_entries_by_category']["Suspicious Keywords"] = keyword_entries
            else:
                return "Column 'Entry Description' not found in the data."

        if app.config['seldomly_used_accounts_var']:
            if app.config['processed_df'] is not None:
                account_frequency = app.config['processed_df']["Account Number"].value_counts().reset_index()
                account_frequency.columns = ["Account Number", "Transaction Count"]
                seldomly_used_accounts = account_frequency[account_frequency["Transaction Count"] < app.config['seldomly_used_accounts_threshold']]
                seldomly_used_entries = app.config['processed_df'][
                    app.config['processed_df']["Account Number"].isin(seldomly_used_accounts["Account Number"])
                ]
                app.config['high_risk_entries'] = pd.concat([app.config['high_risk_entries'], seldomly_used_entries])
                app.config['flagged_entries_by_category']["Seldomly Used Accounts"] = seldomly_used_entries

        if not app.config['high_risk_entries'].empty:
            return f"Found {len(app.config['high_risk_entries'])} high-risk entries."
        else:
            return "No high-risk entries found."
    except Exception as e:
        return f"Error during testing: {e}"

# Function to visualize high-risk entries
def visualize_high_risk_entries():
    if not app.config['high_risk_entries'].empty:
        # Bar chart for counts of high-risk entries by category
        category_counts = {category: len(entries) for category, entries in app.config['flagged_entries_by_category'].items()}
        fig = px.bar(x=list(category_counts.keys()), y=list(category_counts.values()), labels={"x": "Category", "y": "Count"})
        fig.show()

        # Pie chart for distribution of high-risk entries
        fig = px.pie(names=list(category_counts.keys()), values=list(category_counts.values()))
        fig.show()

        # Scatter plot for rounded numbers
        if "Rounded Numbers" in app.config['flagged_entries_by_category']:
            rounded_entries = app.config['flagged_entries_by_category']["Rounded Numbers"]
            fig = px.scatter(rounded_entries, x="Debit Amount (Dr)", y="Credit Amount (Cr)", color="Account Number")
            fig.show()

        # Scatter plot for 99999 pattern
        if "99999 Pattern" in app.config['flagged_entries_by_category']:
            nine_pattern_entries = app.config['flagged_entries_by_category']["99999 Pattern"]
            fig = px.scatter(nine_pattern_entries, x="Debit Amount (Dr)", y="Credit Amount (Cr)", color="Account Number")
            fig.show()

# Authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in authorized_users and authorized_users[username] == password:
            app.config['logged_in'] = True
            app.config['logged_in_user'] = username
            return redirect(url_for('main_app'))
        else:
            flash("Invalid username or password")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/')
def main_app():
    if not app.config['logged_in']:
        return redirect(url_for('login'))
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
