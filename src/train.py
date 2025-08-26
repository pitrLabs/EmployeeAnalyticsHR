from model import HRPredictiveModel

if __name__ == "__main__":
    hr_ml_model = HRPredictiveModel()

    print("=== TRAINING ===")
    ml_model = hr_ml_model.train_turnover_prediction_model()

    if ml_model is not None:
        print("=== FEATURE VALIDATION ===")
        print("Model feature names:", ml_model.feature_names_)

        print("=== PREDICTION ===")
        risk_scores = hr_ml_model.predict_turnover_risk()
        if risk_scores is not None:
            print("\nTop 10 employees with highest turnover risk:")
            print(risk_scores.head(10))

            print(f"\nRisk score statistics:")
            print(f"Min: {risk_scores['turnover_risk'].min():.2f}%")
            print(f"Max: {risk_scores['turnover_risk'].max():.2f}%")
            print(f"Avg: {risk_scores['turnover_risk'].mean():.2f}%")
