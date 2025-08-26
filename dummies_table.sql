CREATE TABLE s_departments (
    department_id UUID PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL,
    location VARCHAR(50),
    budget BIGINT
);

CREATE TABLE s_employees (
    employee_id UUID PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE,
    department_id UUID REFERENCES s_departments(department_id),
    job_title VARCHAR(100),
    hire_date DATE,
    salary BIGINT,
    manager_id UUID,
    status VARCHAR(20)
);

CREATE TABLE s_performance (
    performance_id UUID PRIMARY KEY,
    employee_id UUID REFERENCES s_employees(employee_id),
    review_date DATE,
    rating NUMERIC(2,1),
    reviewer_id UUID,
    comments TEXT
);

CREATE TABLE s_attendance (
    attendance_id UUID PRIMARY KEY,
    employee_id UUID REFERENCES s_employees(employee_id),
    date DATE,
    status VARCHAR(20),
    hours_worked NUMERIC(4,2)
);

CREATE TABLE s_training (
    training_id UUID PRIMARY KEY,
    employee_id UUID REFERENCES s_employees(employee_id),
    training_name VARCHAR(100),
    category VARCHAR(50),
    score NUMERIC(5,2),
    completion_date DATE
);
