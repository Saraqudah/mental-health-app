from flask import Flask, render_template, request  # Flask setup and form handling
import pandas as pd  # for reading Excel files
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.preprocessing import LabelEncoder  # to convert text to numbers
import traceback  # to show errors if something breaks
import sqlite3  # to use SQLite database
import random  # for generating random values
from datetime import datetime  # for working with dates
import os  # to access system info
import smtplib  # to send emails
from email.message import EmailMessage  # to create email messages
from sklearn.model_selection import train_test_split  # to split data for training/testing
from sklearn.metrics import accuracy_score  # to check model accuracy




app = Flask(__name__)# this line creates the Flask app instance so I can start building routes and views

EMAIL_ADDRESS = 'sarayahiaqud5@gmail.com' # the email address I use to send results to users
EMAIL_PASSWORD = 'uykisxuztkyhdsln' # the app password for Gmail (used to log in and send emails securely)

try:
    df = pd.read_excel("data.xlsx")  # loading the dataset from Excel file
    print("✅ Data loaded successfully")

    feature_cols = ['Age', 'Gender', 'Mood Score (1-10)', 'Sleep Quality (1-10)',
                    'Physical Activity (hrs/week)', 'Stress Level (1-10)']  # input features for prediction

    target_cols = ['Diagnosis', 'AI-Detected Emotional State','Outcome', 'Medication', 'Therapy Type']  # what the model will predict

    le_gender = LabelEncoder()  # to convert gender from text to numbers
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    X = df[feature_cols]  # selecting the input columns

    models = {}       # dictionary to store trained models
    accuracies = {}   # dictionary to store accuracy of each model

    print("\n📊 Model accuracy on test data:\n" + "-"*45)
    print(f"{'Target':<30} | {'Accuracy':>10}")
    print("-"*45)

    for target in target_cols:
        y = df[target]  # selecting the current target column
        le = LabelEncoder()  # label encoding for target values
        y_encoded = le.fit_transform(y)

        # splitting data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42)

        # training the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # making predictions and calculating accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # saving the model and its encoder
        models[target] = {
            'model': model,
            'encoder': le,
            'features': feature_cols
        }

        accuracies[target] = accuracy  # storing accuracy
        print(f"{target:<30} | {accuracy:>10.2f}")  # printing accuracy

    print("\n✅ Finished training and testing models.")

except Exception as e:
    print("❌ Error while processing data:")  # if something goes wrong
    traceback.print_exc()  # show error details

def create_table_if_not_exists():
    try:
        conn = sqlite3.connect('mental_health.db', timeout=10)  # connect to SQLite database (or create it if it doesn't exist)
        cursor = conn.cursor()  # create a cursor to run SQL commands
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS database ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,  
                name TEXT, email TEXT, marital_status TEXT, location TEXT, education TEXT,
                age INTEGER, gender TEXT, mood_score INTEGER, sleep_quality INTEGER, 
                physical_activity INTEGER, stress_level INTEGER, 
                diagnosis TEXT, emotion TEXT, outcome TEXT, medication TEXT, therapy TEXT, 
                treatment_start_date TEXT, treatment_duration_weeks INTEGER, 
                adherence REAL, progress REAL 
            )
        ''')  # run the SQL command 
              # create the table only if it doesn't already exist
              
        conn.commit()  # save changes to the database
    except Exception as e:
        print("❌ Error while creating or updating the table:")  # show message if something goes wrong
        traceback.print_exc()  # print the full error details
    finally:
        conn.close()  # always close the database connection

def send_email(to_email, subject, body):
    try:
        msg = EmailMessage()  # create a new email message
        msg['Subject'] = subject  # set the subject of the email
        msg['From'] = EMAIL_ADDRESS  # sender email address
        msg['To'] = to_email  # recipient email address
        msg.set_content(body)  # set the body/content of the email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:  # connect to Gmail's SMTP server securely
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  # login using app email and password
            smtp.send_message(msg)  # send the email
        print(f"📧 Email sent to {to_email}")  # confirmation message
    except Exception:
        print("❌ Failed to send email:")  # error message
        traceback.print_exc()  # show full error details


@app.route('/')  # route for the homepage
def home():
    return render_template('Page0.html')  # show the first page

@app.route('/Page1')  # route for the second page
def page1():
    return render_template('Page1.html')  # show the second page

@app.route('/save_personal_info', methods=['POST'])  # route to save user info
def save_personal_info():
    return render_template('Page2.html',  # go to next page with user data
                           name=request.form['name'],
                           email=request.form['email'],
                           marital_status=request.form['marital_status'],
                           location=request.form['location'],
                           education=request.form['education'])

@app.route('/predict', methods=['POST'])  # route for prediction
def predict():
    try:
        create_table_if_not_exists()  # make sure DB table exists

        age = int(request.form['age'])  # get age from form
        gender = request.form['gender']  # get gender from form
        mood_score = int(request.form['mood_score'])  # get mood score
        sleep_quality = int(request.form['sleep_quality'])  # get sleep quality
        physical_activity = int(request.form['physical_activity'])  # get activity level
        stress_level = int(request.form['stress_level'])  # get stress level

        name = request.form['name']  # get name
        email = request.form['email']  # get email
        marital_status = request.form['marital_status']  # get marital status
        location = request.form['location']  # get location
        education = request.form['education']  # get education

        gender_encoded = le_gender.transform([gender])[0]  # encode gender to (0,1)

        full_input = {  # combine all inputs in one dict
            'Age': age,
            'Gender': gender_encoded,
            'Mood Score (1-10)': mood_score,
            'Sleep Quality (1-10)': sleep_quality,
            'Physical Activity (hrs/week)': physical_activity,
            'Stress Level (1-10)': stress_level
        }

        results = {}  # to store model results
        for target in target_cols:
            model = models[target]['model']  # get trained model
            le = models[target]['encoder']  # get encoder
            features = models[target]['features']  # get feature list
            input_df = pd.DataFrame([{k: full_input[k] for k in features}])  # create input dataframe
            prediction_encoded = model.predict(input_df)[0]  # make prediction
            prediction = le.inverse_transform([prediction_encoded])[0]  # decode result
            results[target] = prediction  # save the result

        conn = sqlite3.connect('mental_health.db', timeout=10)  # connect to database
        cursor = conn.cursor()  # create cursor for SQL
        cursor.execute(""" 
            SELECT mood_score, sleep_quality, stress_level, physical_activity, emotion, treatment_start_date, treatment_duration_weeks
            FROM database WHERE email = ? ORDER BY id DESC LIMIT 1
        """, (email,))
        row = cursor.fetchone()  # fetch the result  # get latest record for this user


        emotion_state = results['AI-Detected Emotional State']
        
        emotion_messages = {
            "Stressed": "ou're feeling Stressed —It's okay to feel stressed sometimes — it's part of being human. Take a deep breath and remind yourself that you're doing your best,and that’s enough. Things will calm down, and you are stronger than you think.🌿\nلا بأس أن تشعر بالتوتر أحيانًا، فهذا جزء من كونك إنسانًا. خذ نفسًا عميقًا، وذكّر نفسك أنك تبذل ما بوسعك، وهذا يكفي. الأمور ستهدأ، وأنت أقوى مما تعتقد.🌿",
            "Happy": "It's wonderful to feel happy! Embrace that joy with gratitude and let your light inspire those around you. Remember, beautiful moments are meant to be fully lived and cherished. Smile — you're spreading positive energy without even trying 😊☀\nجميل أن تشعر بالسعادة! احتضن هذا الشعور بكل امتنان، ودَع نورك يُلهم من حولك. تذكّر أن اللحظات الجميلة تستحق أن نعيشها بوعي وفرح كامل. ابتسم... فأنت تنشر طاقة إيجابية دون أن تدري 😊☀",
            "Anxious": "Anxiety can feel overwhelming, but it doesn’t define you or control you. Remember — thoughts aren’t facts, and this feeling will pass. Take it one moment at a time, and give yourself the peace you deserve. You are safe right now.🕊\nقلق شعور مزعج، لكنه لا يُعرّفك ولا يتحكم بك. تذكّر أن الأفكار ليست حقائق، وأن كل شيء يمر—even هذا القلق. خذ الأمور لحظة بلحظة، وامنح نفسك الطمأنينة التي تستحقها. أنت بأمان الآن. 🕊",
            "Depressed": "I know that feeling depressed can make everything feel heavy... but you're not alone. Your presence matters — even on the days it doesn't feel like it. It's okay to rest, and to give yourself time to heal. One small step is enough today. 🌧💙\nأعلم أن الشعور بالاكتئاب يمكن أن يجعل كل شيء يبدو ثقيلاً... لكنك لست وحدك. وجودك مهم، حتى في الأيام التي لا تشعر فيها بذلك. لا بأس أن تطلب الراحة، وأن تمنح نفسك وقتًا للتعافي. خطوة صغيرة واحدة كافية اليوم. 🌧💙",
            "Excited": "It's amazing that you're feeling excited! That spark means something truly matters to you. Enjoy the moment, and celebrate every step — no matter how small. You're on a path full of possibilities! ⚡🎉\nرائع أنك تشعر بالحماس! هذه الطاقة الجميلة هي علامة على شيء يهمك حقًا. استمتع بكل لحظة، واسمح لنفسك أن تحتفل بخطواتك مهما كانت صغيرة. أنت على طريق مليء بالإمكانيات! ⚡🎉",
            "Neutral": "Feeling calm and connected is a true gift. Savor this balance and be present with it. Like nature, you grow quietly and bloom in your own time. Let this peace guide you. 🍃🌿\nأن تشعر بالسلام الداخلي والانسجام مع نفسك هو نعمة حقيقية. استمتع بهذا الاتزان، وخذ لحظاتك بكل وعي. مثل الطبيعة، أنت تنمو بهدوء، وتزدهر دون استعجال. دَع هذا الهدوء يرشدك. 🍃🌿"
        }

        emotion_message = emotion_messages.get(emotion_state, "")  # get the matching message for the emotion
        is_new_user = row is None  # check if this is a new user


        if is_new_user:
           treatment_start_date = datetime.today().strftime('%Y-%m-%d')  # set treatment start to today
           diagnosis = results['Diagnosis']  # get predicted diagnosis

           # set possible treatment durations for each diagnosis
           duration_ranges = {
            "Major Depressive Disorder": (10, 16),
            "Generalized Anxiety": (10, 18),
            "Bipolar Disorder": (12, 20),
            "Panic Disorder": (8, 14)
            }

           default_range = (8, 12)  # default range if diagnosis not found
           selected_range = duration_ranges.get(diagnosis, default_range)  # pick range for diagnosis
           treatment_duration_weeks = random.randint(*selected_range)  # choose random duration
           adherence = 0  # no adherence for new user
           progress = None  # no progress yet
           weeks_remaining = treatment_duration_weeks  # all weeks still remain
           continue_message = ""  # no message for new user
        else:
           # get saved diagnosis, medication, therapy from first record
          cursor.execute("""
            SELECT diagnosis, medication, therapy
            FROM database
            WHERE email = ?
            ORDER BY treatment_start_date ASC, id ASC 
            LIMIT 1
          """, (email,))
          
          stable_values = cursor.fetchone()
          if stable_values:
            results['Diagnosis'], results['Medication'], results['Therapy Type'] = stable_values

           # unpack old data from last record
          old_mood, old_sleep, old_stress,old_physical, previous_emotion, start_date_str, duration_weeks = row
          treatment_start_date = start_date_str  # use previous start date
          treatment_duration_weeks = duration_weeks  # use saved treatment length
          improved = 0  # counter for improvement points

          if mood_score > old_mood: improved += 1  # mood improved
          if sleep_quality > old_sleep: improved += 1  # sleep improved
          if physical_activity > old_physical: improved += 1 #physical improved
          if stress_level < old_stress: improved += 1  # stress decreased
          

          start_date = datetime.strptime(start_date_str, "%Y-%m-%d")  # convert start date to datetime
          today = datetime.today()  # get current date
          weeks_passed = max(0, (today - start_date).days // 7)  # count how many full weeks passed

          if weeks_passed >= 1:
           # calculate raw progress score
            raw_progress = (mood_score + sleep_quality + physical_activity - stress_level) / 3
            raw_progress = max(0, min(raw_progress, 10))  # limit between 0 and 10

            if treatment_duration_weeks > 0:
                weeks_ratio = min(weeks_passed / treatment_duration_weeks, 1.0)  # percentage of treatment passed
                progress = round(raw_progress * weeks_ratio)  # final progress score
            else:
               progress = 0  # if duration is invalid
          else:
            progress = 0  # not enough time to calculate progress

          improvement_ratio = improved / 4  # ratio of improved metrics
          adherence = improvement_ratio * min(weeks_passed / treatment_duration_weeks, 1.0) * 100 if treatment_duration_weeks else 0  # final adherence %
          weeks_remaining = max(0, treatment_duration_weeks - weeks_passed)  # how many weeks left
          continue_message = "استمر بالعلاج 💪"  # motivational message

          # determine outcome based on emotional change
          positive = ['Happy', 'Excited', 'Neutral']
          negative = ['Stressed', 'Anxious', 'Depressed']
          if previous_emotion in negative and emotion_state in positive:
              results['Outcome'] = 'Improved'  # emotional state got better
          elif previous_emotion in positive and emotion_state in negative:
              results['Outcome'] = 'Deteriorated'  # emotional state got worse
          else:
              results['Outcome'] = 'No Change'  # no major change


        cursor.execute('''  
    INSERT INTO database (
        name, email, marital_status, location, education,
        age, gender, mood_score, sleep_quality, physical_activity, stress_level,
        diagnosis, emotion, outcome, medication, therapy,
        treatment_start_date, treatment_duration_weeks, adherence, progress
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
    (
        name, email, marital_status, location, education,  # personal info
        age, gender, mood_score, sleep_quality, physical_activity, stress_level,  # scores
        results['Diagnosis'], emotion_state, results['Outcome'], results['Medication'], results['Therapy Type'],  # model results
        treatment_start_date, treatment_duration_weeks, adherence, progress  # treatment data
    ))

        conn.commit()  # save changes to the database # insert new user record into the database

        if not is_new_user and weeks_passed >= 1:
          progress_score = round((progress / 10) * 10)  # convert progress to 1–10 scale
          progress_score = max(1, min(progress_score, 10))  # keep score between 1 and 10
         # set progress level label based on score
          progress_level = f"{progress_score}/10 - {'📈 ممتاز' if progress_score >= 8 else '🔄 متوسط' if progress_score >= 5 else '⚠ منخفض'}"
         # set adherence level label
          adherence_level = "✅ عالي" if adherence >= 80 else "🟡 متوسط" if adherence >= 50 else "❌ منخفض"
        elif not is_new_user:
           progress_level = "🛛 لم يمضِ أسبوع على بدء العلاج"  # too early to track progress
           adherence_level = "ℹ لا يمكن حساب الالتزام"  # can't calculate adherence yet
        else:
           progress_level = "لا يوجد سجل سابق لقياس التقدم 💼"  # no data to calculate progress
           adherence_level = "ℹ لم يتم احتساب الالتزام"  # adherence not available

        
        # build email content with prediction results
        email_body = f"""
           Dear {name},

            Thank you for completing your mental health assessment.

           📋 Diagnosis: {results['Diagnosis']}
          🌟 Emotional State: {results['AI-Detected Emotional State']}
          💊 Medication: {results['Medication']}
          🧠 Therapy Type: {results['Therapy Type']}

          📅 Treatment Start Date: {treatment_start_date}
          
         🗓 Treatment Duration: {treatment_duration_weeks} weeks
          
         {f"⏳ Weeks Remaining: {weeks_remaining} weeks" if not is_new_user else ""}

         {"🧾 Outcome: " + results['Outcome'] if not is_new_user else ""}

        Stay well,  
        Your Mental Health Team
        """

        send_email(email, "Your Mental Health Results", email_body.strip())  # send the email


        return render_template("Page3.html",  # render the final result page
          diagnosis=results['Diagnosis'],  # show predicted diagnosis
          emotion=emotion_state,  # show emotional state
          outcome=results['Outcome'] if not is_new_user else None,  # show outcome if not first visit
          medication=results['Medication'],  # show suggested medication
          therapy=results['Therapy Type'],  # show suggested therapy
          emotion_message=emotion_message,  # show supportive message
          progress=progress if not is_new_user else None,  # show progress if available
          adherence=round(adherence, 2) if not is_new_user else None,  # show adherence %
          treatment_start_date=treatment_start_date,  # show treatment start date
          treatment_duration_weeks=treatment_duration_weeks,  # show total treatment length
          is_new_user=is_new_user,  # used to check if this is a new user
          progress_level=progress_level if not is_new_user else None,  # label for progress
          adherence_level=adherence_level if not is_new_user else None,  # label for adherence
          weeks_remaining=weeks_remaining if not is_new_user else None,  # weeks left for treatment
          continue_message=continue_message if not is_new_user else ""  # motivational message
)

    except Exception as e:
           print("❌ خطأ أثناء التنبؤ أو الحفظ:")  # log error message in Arabic
           traceback.print_exc()  # show full error details
           return "حدث خطأ أثناء المعالجة. تحقق من المدخلات."  # return error message to user
    finally:
       try:
        conn.close()  # close the database connection
       except:
         pass  # ignore any close errors


if __name__ == '__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)