import mysql.connector
import uuid
import random
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from decouple import config
from faker import Faker

# conn = mysql.connector.connect(host=config("MSQL_HOST"),
#                         database=config("MSQL_DB"),
#                         user=config("MSQL_USER"),
#                         password=config("MSQL_PASS"),
#                         port=config("MSQL_PORT"))

conn = psycopg2.connect(host=config("PSQL_HOST2"),
                        database=config("PSQL_DB2"),
                        user=config("PSQL_USER2"),
                        password=config("PSQL_PASS2"),
                        port=config("PSQL_PORT2"))

fake = Faker('id_ID')


def create_sample_employees(n):
    departments_query = "SELECT department_id, department_name FROM s_departments"
    departments_df = pd.read_sql_query(departments_query, conn)

    if departments_df.empty:
        pass

    else:
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


def create_sample_performance():
    employees_query = """
        SELECT employee_id, department_id, manager_id, job_title, hire_date, status 
        FROM s_employees 
        WHERE status = 'active' 
          AND (manager_id IS NOT NULL)
    """
    employees_df = pd.read_sql_query(employees_query, conn)
    data = []
    current_date = datetime.now()

    for _, row in employees_df.iterrows():
        emp_id = row['employee_id']
        hire_date = pd.to_datetime(row['hire_date'])
        manager_id = row['manager_id']

        for year_offset in range(1, 4):
            review_date = hire_date + relativedelta(years=year_offset)

            if review_date < current_date:
                review = {
                    'performance_id': str(uuid.uuid4()),
                    'employee_id': emp_id,
                    'review_date': review_date,
                    'rating': round(random.uniform(3.0, 5.0), 1),
                    'reviewer_id': manager_id,
                    'comments': f'Performance review {review_date.year}'
                }
                data.append(review)

    return pd.DataFrame(data)


def create_sample_attendance(days):
    employees_query = "SELECT employee_id, hire_date, status FROM s_employees WHERE status = 'active'"
    employees_df = pd.read_sql_query(employees_query, conn)

    data = []
    today = datetime.now()

    for _, row in employees_df.iterrows():
        emp_id = row['employee_id']
        hire_date = pd.to_datetime(row['hire_date'])
        emp_start_date = max(hire_date, today - timedelta(days=days))

        for day_offset in range((today - emp_start_date).days):
            date = emp_start_date + timedelta(days=day_offset)

            if date.weekday() < 5:
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

                data.append({
                    'attendance_id': str(uuid.uuid4()),
                    'employee_id': emp_id,
                    'date': date,
                    'status': status,
                    'hours_worked': round(hours, 2)
                })

    return pd.DataFrame(data)


def create_sample_training():
    employees_query = """
        SELECT employee_id, department_id, manager_id, job_title, hire_date, status 
        FROM s_employees 
        WHERE status = 'active'
    """
    employees_df = pd.read_sql_query(employees_query, conn)

    employees_df['hire_date'] = pd.to_datetime(employees_df['hire_date'],
                                               errors='coerce',
                                               format='%Y-%m-%d')

    job_training_map = {
        'Software Engineer': ['Coding', 'Data Analysis'],
        'Data Analyst': ['Data Analysis', 'Coding'],
        'System Admin': ['Coding', 'Project Management'],
        'IT Manager': ['Leadership Workshop', 'Project Management'],

        'HR Specialist': ['Communication Skills', 'Project Management'],
        'Recruiter': ['Communication Skills', 'Leadership Workshop'],
        'HR Manager': ['Leadership Workshop', 'Project Management'],
        'Training Coordinator': ['Communication Skills', 'Project Management'],

        'Accountant': ['Data Analysis', 'Project Management'],
        'Financial Analyst': ['Data Analysis', 'Coding'],
        'Finance Manager': ['Leadership Workshop', 'Project Management'],
        'Auditor': ['Data Analysis', 'Communication Skills'],

        'Marketing Specialist': ['Communication Skills', 'Project Management'],
        'Content Creator': ['Communication Skills'],
        'Marketing Manager': ['Leadership Workshop', 'Project Management'],
        'SEO Analyst': ['Data Analysis', 'Communication Skills'],

        'Operations Manager': ['Leadership Workshop', 'Project Management'],
        'Logistics Coordinator': ['Project Management', 'Communication Skills'],
        'Supply Chain Specialist': ['Data Analysis', 'Project Management'],

        'Sales Representative': ['Communication Skills'],
        'Account Manager': ['Project Management', 'Communication Skills'],
        'Sales Manager': ['Leadership Workshop', 'Project Management'],
        'Business Development': ['Leadership Workshop', 'Communication Skills']
    }

    training_info = {
        'Leadership Workshop': 'Soft Skills',
        'Coding': 'Technical',
        'Data Analysis': 'Technical',
        'Communication Skills': 'Soft Skills',
        'Project Management': 'Management'
    }

    data = []

    for _, row in employees_df.iterrows():
        emp_id = row['employee_id']
        job_title = row['job_title']
        hire_date = row['hire_date']
        manager_id = row['manager_id']

        if pd.isna(hire_date) or job_title not in job_training_map:
            continue

        trainings = job_training_map[job_title]

        for t in trainings:
            if t == 'Project Management' and pd.notna(manager_id):
                continue

            participant = {
                'training_id': str(uuid.uuid4()),
                'employee_id': emp_id,
                'training_name': t,
                'category': training_info[t],
                'score': round(random.uniform(60, 100), 2),
                'completion_date': hire_date + relativedelta(months=6)
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
    performance_df = create_sample_performance()
    attendance_df = create_sample_attendance(90)
    training_df = create_sample_training()

    # load_to_postgres(departments_df, "s_departments", conn)
    # load_to_postgres(employees_df, "s_employees", conn)
    # load_to_postgres(performance_df, "s_performance", conn)
    # load_to_postgres(attendance_df, "s_attendance", conn)
    load_to_postgres(training_df, "s_training", conn)

    conn.close()
