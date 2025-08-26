import uuid
import random
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from decouple import config
from faker import Faker

conn = psycopg2.connect(host=config("PSQL_HOST"),
                        database=config("PSQL_DB"),
                        user=config("PSQL_USER"),
                        password=config("PSQL_PASS"),
                        port=config("PSQL_PORT"))
fake = Faker('id_ID')


def create_sample_employees(n):
    departments_query = "SELECT department_id, department_name FROM s_departments"
    departments_df = pd.read_sql_query(departments_query, conn)

    if departments_df.empty:
        raise ValueError("Tidak ada data departemen ditemukan dalam database")

    job_titles = {
        'IT': ['Software Engineer', 'Data Analyst', 'System Admin', 'IT Manager'],
        'HR': ['HR Specialist', 'Recruiter', 'HR Manager', 'Training Coordinator'],
        'Finance': ['Accountant', 'Financial Analyst', 'Finance Manager', 'Auditor'],
        'Marketing': ['Marketing Specialist', 'Content Creator', 'Marketing Manager', 'SEO Analyst'],
        'Operations': ['Operations Manager', 'Logistics Coordinator', 'Supply Chain Specialist'],
        'Sales': ['Sales Representative', 'Account Manager', 'Sales Manager', 'Business Development']
    }

    data = []
    start_date = datetime(2018, 1, 1)
    departments = departments_df['department_name'].tolist()
    managers_by_dept = {d: [] for d in departments}

    for dept in departments:
        available_titles = job_titles[dept]
        mgr_titles = [t for t in available_titles if 'Manager' in t]

        mgr_id = str(uuid.uuid4())
        hire_date = start_date + timedelta(days=random.randint(0, 500))
        dept_row = departments_df.loc[departments_df['department_name'] == dept]

        if dept_row.empty:
            continue

        dept_id = dept_row['department_id'].iloc[0]

        manager = {
            'employee_id': mgr_id,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': f"{fake.user_name()}@example.co.id",
            'department_id': dept_id,
            'job_title': mgr_titles[0],
            'hire_date': hire_date,
            'salary': random.randint(80000000, 200000000),
            'manager_id': None,
            'status': 'active'
        }
        data.append(manager)
        managers_by_dept[dept].append(mgr_id)

    managers_count = sum(len(v) for v in managers_by_dept.values())
    remaining = max(0, n - managers_count)

    for _ in range(remaining):
        dept_row = departments_df.sample(1).iloc[0]
        dept = dept_row['department_name']
        dept_id = dept_row['department_id']

        available_titles = job_titles[dept]
        mgr_titles = [t for t in available_titles if
                      t in ['IT Manager', 'HR Manager', 'Finance Manager',
                            'Marketing Manager', 'Operations Manager', 'Sales Manager']]
        non_mgr_titles = [t for t in available_titles if
                          t not in ['IT Manager', 'HR Manager', 'Finance Manager',
                                    'Marketing Manager', 'Operations Manager', 'Sales Manager']]

        job = random.choice(non_mgr_titles)
        emp_id = str(uuid.uuid4())
        hire_date = start_date + timedelta(days=random.randint(0, 1000))
        manager_id = random.choice(managers_by_dept[dept]) if managers_by_dept[dept] else None

        employee = {
            'employee_id': emp_id,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': f"{fake.user_name()}@example.co.id",
            'department_id': dept_id,
            'job_title': job,
            'hire_date': hire_date,
            'salary': random.randint(30000000, 120000000),
            'manager_id': manager_id,
            'status': random.choices(['active', 'resigned', 'terminated'],
                                     weights=[0.8, 0.15, 0.05])[0]
        }
        data.append(employee)

    return pd.DataFrame(data)


def create_sample_departments():
    data = [
        {'department_id': str(uuid.uuid4()), 'department_name': 'IT', 'location': 'Jakarta', 'budget': 5000000000},
        {'department_id': str(uuid.uuid4()), 'department_name': 'HR', 'location': 'Jakarta', 'budget': 2000000000},
        {'department_id': str(uuid.uuid4()), 'department_name': 'Finance', 'location': 'Jakarta', 'budget': 3000000000},
        {'department_id': str(uuid.uuid4()), 'department_name': 'Marketing', 'location': 'Jakarta', 'budget': 4000000000},
        {'department_id': str(uuid.uuid4()), 'department_name': 'Operations', 'location': 'Bandung', 'budget': 3500000000},
        {'department_id': str(uuid.uuid4()), 'department_name': 'Sales', 'location': 'Jakarta', 'budget': 4500000000}
    ]
    return pd.DataFrame(data)


def create_sample_performance(employees_df):
    data = []
    current_date = datetime.now()

    for emp_id in employees_df['employee_id']:
        for year in [2022, 2023, 2024]:
            review_date = datetime(year, random.randint(1, 12), random.randint(1, 28))

            if review_date < current_date:
                review = {
                    'performance_id': str(uuid.uuid4()),
                    'employee_id': emp_id,
                    'review_date': review_date,
                    'rating': round(random.uniform(3.0, 5.0), 1),
                    'reviewer_id': str(uuid.uuid4()),
                    'comments': f'Performance review for {year}'
                }
                data.append(review)

    return pd.DataFrame(data)


def create_sample_attendance(employees_df, days=90):
    data = []
    start_date = datetime.now() - timedelta(days=days)

    for emp_id in employees_df[employees_df['status'] == 'active']['employee_id']:
        for day in range(days):
            date = start_date + timedelta(days=day)

            if date.weekday() < 5:  # Only weekdays
                status = random.choices(['present', 'absent', 'late', 'early_departure'],
                                        weights=[0.85, 0.05, 0.07, 0.03])[0]

                if status == 'present':
                    hours = random.uniform(7.5, 9.5)
                elif status == 'late':
                    hours = random.uniform(6.0, 8.0)
                elif status == 'early_departure':
                    hours = random.uniform(5.0, 7.0)
                else:
                    hours = 0

                attendance = {
                    'attendance_id': str(uuid.uuid4()),
                    'employee_id': emp_id,
                    'date': date,
                    'status': status,
                    'hours_worked': round(hours, 2)
                }
                data.append(attendance)

    return pd.DataFrame(data)


def create_sample_training(employees_df):
    trainings = [
        {'name': 'Leadership Workshop', 'category': 'Soft Skills'},
        {'name': 'Coding', 'category': 'Technical'},
        {'name': 'Data Analysis', 'category': 'Technical'},
        {'name': 'Communication Skills', 'category': 'Soft Skills'},
        {'name': 'Project Management', 'category': 'Management'}
    ]

    data = []
    for training in trainings:
        participants = random.sample(list(employees_df['employee_id']), random.randint(15, 30))
        for emp_id in participants:
            participant = {
                'training_id': str(uuid.uuid4()),
                'employee_id': emp_id,
                'training_name': training['name'],
                'category': training['category'],
                'score': round(random.uniform(60, 100), 2),
                'completion_date': datetime(2024, random.randint(1, 12), random.randint(1, 28))
            }
            data.append(participant)

    return pd.DataFrame(data)


def load_to_postgres(df, table_name, conn):
    cur = conn.cursor()

    for _, row in df.iterrows():
        cols = ','.join(row.index)
        vals = [row[col] for col in row.index]
        placeholders = ','.join(['%s'] * len(vals))
        query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        cur.execute(query, vals)

    conn.commit()
    cur.close()


if __name__ == "__main__":
    departments_df = create_sample_departments()
    employees_df = create_sample_employees(200)
    performance_df = create_sample_performance(employees_df)
    attendance_df = create_sample_attendance(employees_df, days=90)
    training_df = create_sample_training(employees_df)

    # load_to_postgres(departments_df, "s_departments", conn)
    # load_to_postgres(employees_df, "s_employees", conn)
    # load_to_postgres(performance_df, "s_performance", conn)
    # load_to_postgres(attendance_df, "s_attendance", conn)
    # load_to_postgres(training_df, "s_training", conn)

    conn.close()
