from model import HRAnalyticsModel

if __name__ == "__main__":
    hr_model = HRAnalyticsModel()

    report = hr_model.generate_comprehensive_report()

    print("=== HR ANALYTICS REPORT ===")
    print(f"Total Employees: {report['employee_summary']['total_employees']}")
    print(f"Active Employees: {report['employee_summary']['active_employees']}")
    print(f"Turnover Rate: {report['turnover_rate']}%")
    print(f"\nAverage Salary: {report['employee_summary']['average_salary']}")
    print(f"Average Tenure: {report['employee_summary']['average_tenure']}")

    print("\n=== DEPARTMENT PERFORMANCE ===")
    print(report['department_performance'])

    print("\n=== ATTENDANCE STATS ===")
    for key, value in report['attendance_stats'].items():
        print(f"{key}: {value}")

    # Visualisasi
    hr_model.plot_turnover_by_department()
    hr_model.plot_salary_distribution()
    hr_model.plot_performance_trends()
    hr_model.plot_attendance_patterns()

    # Cek sample data
    print("\nSample of employees data:")
    print(hr_model.employees_df.head())
    print(f"\nShape of attendance data: {hr_model.attendance_df.shape}")
