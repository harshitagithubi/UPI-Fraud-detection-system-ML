
from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import sqlite3
from pathlib import Path

app = Flask(__name__)

# Project directory
BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "transactions.db"
MODEL_PATH = BASE_DIR / "random_forest_pipeline.pkl"

# Load improved Random Forest pipeline
model = joblib.load(MODEL_PATH)


# 1. Database connection
def get_db_connection():
    return sqlite3.connect(DATABASE)


# 2. Initialise and migrate database
def init_db():

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            trans_hour INTEGER,
            trans_day INTEGER,
            trans_month INTEGER,
            trans_amount REAL,
            age INTEGER,
            category INTEGER,
            state INTEGER,
            prediction TEXT
        )
    """)

    # Check existing database columns
    c.execute("PRAGMA table_info(transactions)")
    columns = [row[1] for row in c.fetchall()]

    old_columns = {
        "trans_year",
        "upi_number"
    }

    if old_columns.intersection(columns):

        # Preserve the old table before migrating
        c.execute("""
            ALTER TABLE transactions
            RENAME TO transactions_old
        """)

        c.execute("""
            CREATE TABLE transactions (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                trans_hour INTEGER,
                trans_day INTEGER,
                trans_month INTEGER,
                trans_amount REAL,
                age INTEGER,
                category INTEGER,
                state INTEGER,
                prediction TEXT
            )
        """)

        # Copy existing records.
        # New feature values are NULL for old records.
        c.execute("""
            INSERT INTO transactions
            (Id, trans_hour, trans_day, trans_month,
             trans_amount, prediction)
            SELECT Id, trans_hour, trans_day, trans_month,
                   trans_amount, prediction
            FROM transactions_old
        """)

    conn.commit()
    conn.close()


init_db()


# 3. Homepage
@app.route('/')
def home():
    return render_template('index.html')


# 4. Collect and validate transaction input
def get_transaction_input(form_data):

    trans_hour = int(form_data['trans_hour'])
    trans_day = int(form_data['trans_day'])
    trans_month = int(form_data['trans_month'])
    trans_amount = float(form_data['trans_amount'])
    age = int(form_data['age'])
    category = int(form_data['category'])
    state = int(form_data['state'])

    if not 0 <= trans_hour <= 23:
        raise ValueError("Transaction hour must be between 0 and 23.")

    if not 1 <= trans_day <= 31:
        raise ValueError("Transaction day must be between 1 and 31.")

    if not 1 <= trans_month <= 12:
        raise ValueError("Transaction month must be between 1 and 12.")

    if trans_amount <= 0:
        raise ValueError("Transaction amount must be positive.")

    if not 14 <= age <= 94:
        raise ValueError("Age must be between 14 and 94.")

    if not 0 <= category <= 13:
        raise ValueError("Invalid category code.")

    if not 0 <= state <= 50:
        raise ValueError("Invalid state code.")

    return {
        "trans_hour": trans_hour,
        "trans_day": trans_day,
        "trans_month": trans_month,
        "trans_amount": trans_amount,
        "age": age,
        "category": category,
        "state": state
    }


# 5. Make prediction
def predict_transaction(input_data):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return "Fraud" if prediction == 1 else "Not Fraud"


# 6. Predict and save transaction
@app.route('/predict', methods=['POST'])
def predict():

    try:
        input_data = get_transaction_input(request.form)
        result = predict_transaction(input_data)

        action = request.form.get('action')

        if action == 'submit':

            conn = get_db_connection()
            c = conn.cursor()

            c.execute("""
                INSERT INTO transactions
                (trans_hour, trans_day, trans_month,
                 trans_amount, age, category, state, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                input_data["trans_hour"],
                input_data["trans_day"],
                input_data["trans_month"],
                input_data["trans_amount"],
                input_data["age"],
                input_data["category"],
                input_data["state"],
                result
            ))

            conn.commit()
            conn.close()

        elif action != 'predict':
            return "Invalid action.", 400

        return render_template('index.html', result=result)

    except (ValueError, TypeError):
        return "Please enter valid transaction details.", 400


# 7. Transaction history
@app.route('/history')
def history():

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
        SELECT Id, trans_hour, trans_day, trans_month,
               trans_amount, age, category, state, prediction
        FROM transactions
        ORDER BY Id DESC
    """)

    records = c.fetchall()
    conn.close()

    return render_template('history.html', records=records)


# 8. Delete transaction
@app.route('/delete/<int:id>')
def delete(id):

    conn = get_db_connection()
    c = conn.cursor()

    c.execute(
        "DELETE FROM transactions WHERE Id = ?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for('history'))


# 9. Edit transaction
@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):

    conn = get_db_connection()
    c = conn.cursor()

    if request.method == 'POST':

        try:
            input_data = get_transaction_input(request.form)

            # Recalculate prediction after editing
            result = predict_transaction(input_data)

            c.execute("""
                UPDATE transactions
                SET trans_hour = ?,
                    trans_day = ?,
                    trans_month = ?,
                    trans_amount = ?,
                    age = ?,
                    category = ?,
                    state = ?,
                    prediction = ?
                WHERE Id = ?
            """, (
                input_data["trans_hour"],
                input_data["trans_day"],
                input_data["trans_month"],
                input_data["trans_amount"],
                input_data["age"],
                input_data["category"],
                input_data["state"],
                result,
                id
            ))

            conn.commit()
            conn.close()

            return redirect(url_for('history'))

        except (ValueError, TypeError):
            conn.close()
            return "Please enter valid transaction details.", 400

    c.execute("""
        SELECT Id, trans_hour, trans_day, trans_month,
               trans_amount, age, category, state, prediction
        FROM transactions
        WHERE Id = ?
    """, (id,))

    record = c.fetchone()
    conn.close()

    if record is None:
        return "Transaction not found.", 404

    return render_template('edit.html', record=record)


if __name__ == '__main__':
    app.run(debug=True)
