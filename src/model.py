import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import traceback
from datetime import datetime, timedelta
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fake = Faker()


class HRAnalyticsModel:
    def __init__(self):
        self.employees_df = self.create_sample_employees()
        self.departments_df = self.create_sample_departments()
        self.performance_df = self.create_sample_performance()
        self.attendance_df = self.create_sample_attendance()
        self.training_df = self.create_sample_training()

    @staticmethod
    def create_sample_employees(n=100):
        departments = ['IT', 'HR', 'Finance', 'Marketing', 'Operations', 'Sales']
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

        for i in range(n):
            dept = random.choice(departments)
            job = random.choice(job_titles[dept])
            hire_date = start_date + timedelta(days=random.randint(0, 1000))

            employee = {
                'employee_id': i + 1,
                'first_name': f'{fake.first_name()}',
                'last_name': f'{fake.last_name()}',
                'email': f'employee{i + 1}@company.com',
                'department': dept,
                'job_title': job,
                'hire_date': hire_date,
                'salary': random.randint(30000000, 120000000),
                'manager_id': random.randint(1, 10) if i > 9 else None,
                'status': random.choices(['active', 'resigned', 'terminated'],
                                         weights=[0.8, 0.15, 0.05])[0]
            }
            data.append(employee)

        return pd.DataFrame(data)

    @staticmethod
    def create_sample_departments():
        data = [
            {'department_id': 1, 'department_name': 'IT', 'location': 'Jakarta', 'budget': 5000000000},
            {'department_id': 2, 'department_name': 'HR', 'location': 'Jakarta', 'budget': 2000000000},
            {'department_id': 3, 'department_name': 'Finance', 'location': 'Jakarta', 'budget': 3000000000},
            {'department_id': 4, 'department_name': 'Marketing', 'location': 'Jakarta', 'budget': 4000000000},
            {'department_id': 5, 'department_name': 'Operations', 'location': 'Bandung', 'budget': 3500000000},
            {'department_id': 6, 'department_name': 'Sales', 'location': 'Jakarta', 'budget': 4500000000}
        ]
        return pd.DataFrame(data)

    def create_sample_performance(self):
        data = []
        current_date = datetime.now()

        for emp_id in self.employees_df['employee_id']:
            for year in [2022, 2023, 2024]:
                review_date = datetime(year, random.randint(1, 12), random.randint(1, 28))
                if review_date < current_date:
                    review = {
                        'employee_id': emp_id,
                        'review_date': review_date,
                        'rating': round(random.uniform(3.0, 5.0), 1),
                        'reviewer_id': random.randint(1, 10),
                        'comments': f'Performance review for {year}'
                    }
                    data.append(review)

        return pd.DataFrame(data)

    def create_sample_attendance(self, days=90):
        data = []
        start_date = datetime.now() - timedelta(days=days)

        for emp_id in self.employees_df[self.employees_df['status'] == 'active']['employee_id']:
            for day in range(days):
                date = start_date + timedelta(days=day)
                if date.weekday() < 5:
                    status = random.choices(
                        ['present', 'absent', 'late', 'early_departure'],
                        weights=[0.85, 0.05, 0.07, 0.03]
                    )[0]

                    if status == 'present':
                        hours = random.uniform(7.5, 9.5)
                    elif status == 'late':
                        hours = random.uniform(6.0, 8.0)
                    elif status == 'early_departure':
                        hours = random.uniform(5.0, 7.0)
                    else:
                        hours = 0

                    data.append({
                        'employee_id': emp_id,
                        'date': date,
                        'status': status,
                        'hours_worked': round(hours, 2)
                    })

        return pd.DataFrame(data)

    def create_sample_training(self):
        trainings = [
            {'training_id': 1, 'name': 'Leadership Workshop', 'category': 'Soft Skills'},
            {'training_id': 2, 'name': 'Coding', 'category': 'Technical'},
            {'training_id': 3, 'name': 'Data Analysis', 'category': 'Technical'},
            {'training_id': 4, 'name': 'Communication Skills', 'category': 'Soft Skills'},
            {'training_id': 5, 'name': 'Project Management', 'category': 'Management'}
        ]
        data = []
        for training in trainings:
            participants = random.sample(list(self.employees_df['employee_id']),
                                         random.randint(15, 30))
            for emp_id in participants:
                data.append({
                    'employee_id': emp_id,
                    'training_id': training['training_id'],
                    'training_name': training['name'],
                    'category': training['category'],
                    'score': round(random.uniform(60, 100), 2),
                    'completion_date': datetime(2024, random.randint(1, 12), random.randint(1, 28))
                })
        return pd.DataFrame(data)

    # analytics, visualization, report methods (sama persis dengan file asli) ...


class HRPredictiveModel(HRAnalyticsModel):
    def __init__(self):
        super().__init__()
        self.ml_models = {}

    def prepare_features_for_ml(self):
        features_df = self.employees_df.copy()

        # Feature engineering - make sure handle missing values
        features_df['tenure'] = (datetime.now() - pd.to_datetime(features_df['hire_date'])).dt.days / 365.25
        features_df['tenure'] = features_df['tenure'].fillna(0)

        features_df['salary_normalized'] = features_df['salary'] / features_df['salary'].max()
        features_df['salary_normalized'] = features_df['salary_normalized'].fillna(0)

        # Aggregate performance data
        perf_agg = self.performance_df.groupby('employee_id')['rating'].agg(['mean', 'std', 'count']).reset_index()
        perf_agg.columns = ['employee_id', 'performance_mean', 'performance_std', 'performance_count']

        features_df = features_df.merge(perf_agg, on='employee_id', how='left')

        # Fill missing performance values
        features_df['performance_mean'] = features_df['performance_mean'].fillna(
            features_df['performance_mean'].median())
        features_df['performance_std'] = features_df['performance_std'].fillna(0)
        features_df['performance_count'] = features_df['performance_count'].fillna(0)

        # Aggregate attendance data
        if not self.attendance_df.empty:
            attendance_agg = self.attendance_df.groupby('employee_id').agg({
                'hours_worked': 'mean',
                'status': lambda x: (x == 'absent').mean()  # absence rate
            }).reset_index()
            attendance_agg.columns = ['employee_id', 'avg_hours_worked', 'absence_rate']
            features_df = features_df.merge(attendance_agg, on='employee_id', how='left')

            # Fill missing attendance values
            features_df['avg_hours_worked'] = features_df['avg_hours_worked'].fillna(
                features_df['avg_hours_worked'].median())
            features_df['absence_rate'] = features_df['absence_rate'].fillna(0)

        else:
            features_df['avg_hours_worked'] = 8.0  # default value
            features_df['absence_rate'] = 0.0

        # Convert categorical variables - make sure existing columns
        if 'department' in features_df.columns:
            features_df = pd.get_dummies(features_df, columns=['department'], prefix='dept')

        if 'job_title' in features_df.columns:
            features_df = pd.get_dummies(features_df, columns=['job_title'], prefix='job')

        return features_df

    def create_turnover_prediction_dataset(self):
        features_df = self.prepare_features_for_ml()

        if 'status' not in features_df.columns:
            print("Error: 'status' column not found in features_df")
            return None, None

        y = (features_df['status'] != 'active').astype(int)
        print(f"Label value counts: {y.value_counts().to_dict()}")
        columns_to_drop = ['employee_id', 'first_name', 'last_name', 'email', 'hire_date', 'status']
        columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
        x = features_df.drop(columns_to_drop, axis=1, errors='ignore')

        if 'will_resign' in x.columns:
            x = x.drop('will_resign', axis=1)

        print(f"X columns: {x.columns.tolist()}")
        print(f"X shape: {x.shape}, y shape: {y.shape}")

        return x, y

    def train_turnover_prediction_model(self):
        try:
            x, y = self.create_turnover_prediction_dataset()

            if x is None or y is None:
                print("Failed to create dataset")

                return None

            feature_names = x.columns.tolist()
            print(f"Feature names for model: {feature_names}")

            imputer = SimpleImputer(strategy='median')
            x_imputed = imputer.fit_transform(x)
            x_train, x_test, y_train, y_test = train_test_split(x_imputed,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=y)

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            print("Model Evaluation:")
            print(classification_report(y_test, y_pred))

            model.feature_names_ = feature_names

            self.ml_models['turnover_prediction'] = model
            self.ml_models['imputer'] = imputer
            self.ml_models['feature_names'] = feature_names

            print(f"Model trained successfully with {len(feature_names)} features")

            return model

        except Exception as e:
            print(f"Error training model: {e}")
            traceback.print_exc()

            return None

    def predict_turnover_risk(self, employee_id=None):
        try:
            if 'turnover_prediction' not in self.ml_models:
                print("Training model first...")
                self.train_turnover_prediction_model()

                if 'turnover_prediction' not in self.ml_models:
                    return None

            features_df = self.prepare_features_for_ml()

            # Ensure to drop several duplicate columns when training
            columns_to_drop = ['employee_id', 'first_name', 'last_name', 'email', 'hire_date', 'status']
            columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]

            x_all = features_df.drop(columns_to_drop, axis=1, errors='ignore')

            if 'will_resign' in x_all.columns:
                x_all = x_all.drop('will_resign', axis=1)

            feature_names = self.ml_models['feature_names']
            print(f"Expected features: {feature_names}")
            print(f"Available features: {x_all.columns.tolist()}")

            missing_features = set(feature_names) - set(x_all.columns)
            extra_features = set(x_all.columns) - set(feature_names)

            if missing_features:
                print(f"Warning: Missing features {missing_features}, filling with 0")

                for feature in missing_features:
                    x_all[feature] = 0

            if extra_features:
                print(f"Warning: Dropping extra features {extra_features}")
                x_all = x_all.drop(list(extra_features), axis=1)

            x_all = x_all[feature_names]

            # Impute missing values
            x_all_imputed = self.ml_models['imputer'].transform(x_all)

            predictions = self.ml_models['turnover_prediction'].predict_proba(x_all_imputed)[:, 1]
            results = features_df[['employee_id', 'first_name', 'last_name']].copy()

            # Add department if needed
            if 'department' in features_df.columns:
                results['department'] = features_df['department']

            elif 'dept_IT' in features_df.columns:
                dept_columns = [col for col in features_df.columns if col.startswith('dept_')]

                if dept_columns:
                    results['department'] = features_df[dept_columns].idxmax(axis=1).str.replace('dept_', '')

            results['turnover_risk'] = (predictions * 100).round(2)
            results['status'] = features_df['status']

            return results.sort_values('turnover_risk', ascending=False)

        except Exception as e:
            print(f"Error predicting turnover risk: {e}")
            traceback.print_exc()

            return None

    def analyze_feature_importance(self):
        try:
            if 'turnover_prediction' not in self.ml_models:
                self.train_turnover_prediction_model()

            model = self.ml_models['turnover_prediction']
            x, _ = self.create_turnover_prediction_dataset()

            feature_importances = pd.DataFrame({
                'feature': model.feature_names_,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importances.head(10), x='importance', y='feature')
            plt.title('Top 10 Features Affecting Employee Turnover')
            plt.tight_layout()
            plt.show()

            return feature_importances

        except Exception as e:
            print(f"Error analyzing feature importance: {e}")

            return None