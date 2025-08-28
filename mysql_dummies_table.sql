CREATE TABLE s_departments (
    department_id CHAR(36) PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL,
    location VARCHAR(50),
    budget BIGINT
);

CREATE TABLE s_employees (
    employee_id CHAR(36) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) UNIQUE,
    department_id CHAR(36),
    job_title VARCHAR(100),
    hire_date DATE,
    salary BIGINT,
    manager_id CHAR(36),
    status VARCHAR(20),
    FOREIGN KEY (department_id) REFERENCES s_departments(department_id),
    FOREIGN KEY (manager_id) REFERENCES s_employees(employee_id)
);

CREATE TABLE s_performance (
    performance_id CHAR(36) PRIMARY KEY,
    employee_id CHAR(36),
    review_date DATE,
    rating DECIMAL(2,1),
    reviewer_id CHAR(36),
    comments TEXT,
    FOREIGN KEY (employee_id) REFERENCES s_employees(employee_id),
    FOREIGN KEY (reviewer_id) REFERENCES s_employees(employee_id)
);

CREATE TABLE s_attendance (
    attendance_id CHAR(36) PRIMARY KEY,
    employee_id CHAR(36),
    date DATE,
    status VARCHAR(20),
    hours_worked DECIMAL(4,2),
    FOREIGN KEY (employee_id) REFERENCES s_employees(employee_id)
);

CREATE TABLE s_training (
    training_id CHAR(36) PRIMARY KEY,
    employee_id CHAR(36),
    training_name VARCHAR(100),
    category VARCHAR(50),
    score DECIMAL(5,2),
    completion_date DATE,
    FOREIGN KEY (employee_id) REFERENCES s_employees(employee_id)
);
