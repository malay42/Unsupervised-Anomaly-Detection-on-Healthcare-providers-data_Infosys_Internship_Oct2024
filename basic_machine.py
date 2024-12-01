import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class OutlierDetector:
    def __init__(self, data, best_params=None):
        default_params = {
            'n_estimators': 52,
            'max_samples': 0.8022683632977241,
            'contamination': 0.05021056937750919,
            'max_features': 0.541298046531514
        }
        
        self.params = best_params or default_params
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data)
        
        self.model = IsolationForest(
            n_estimators=int(self.params['n_estimators']),
            max_samples=self.params['max_samples'],
            contamination=self.params['contamination'],
            max_features=self.params['max_features'],
            random_state=42
        )
        self.model.fit(X_scaled)
        
        # Store original data for visualization
        self.original_data = data
        self.numeric_columns = data.columns.tolist()
    
    def predict_outlier(self, sample):
        if isinstance(sample, pd.Series):
            sample = sample.values.reshape(1, -1)
        elif isinstance(sample, list):
            sample = np.array(sample).reshape(1, -1)
        elif isinstance(sample, np.ndarray) and sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        sample_scaled = self.scaler.transform(sample)
        prediction = self.model.predict(sample_scaled)[0]
        anomaly_score = self.model.score_samples(sample_scaled)[0]
        
        return {
            'is_outlier': prediction == -1,
            'anomaly_score': float(anomaly_score)
        }
    
    def visualize_row(self, row_index, save_path=None):
        """Visualize the specified row in context of all data"""
        row = self.original_data.iloc[row_index]
        detection_result = self.predict_outlier(row)
        
        # Select first two numeric columns for visualization
        cols_to_plot = self.numeric_columns[:2]
        
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.original_data[cols_to_plot])
        plt.title('Data Distribution')
        plt.xticks(rotation=45)
        
        # Subplot 2: Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(
            self.original_data[cols_to_plot[0]], 
            self.original_data[cols_to_plot[1]], 
            alpha=0.5, 
            label='All Data'
        )
        plt.scatter(
            row[cols_to_plot[0]], 
            row[cols_to_plot[1]], 
            color='red', 
            s=100, 
            label='Current Row'
        )
        plt.xlabel(cols_to_plot[0])
        plt.ylabel(cols_to_plot[1])
        plt.title(f'Outlier Status: {detection_result["is_outlier"]}')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return detection_result

def interactive_outlier_detection():
    # Load your data
    data = pd.read_csv('cleaned2_encoded.csv')
    
    # Remove non-numeric columns 
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    X = data[numeric_columns]
    
    # Create outlier detector
    detector = OutlierDetector(X)
    
    while True:
        print("\n--- OUTLIERS DETECTION OPTIONS ---")
        print("1. CHECK SINGLE ROW")
        print("2. CHECK MULTIPLE ROW")
        print("3. VISUALIZE ROW")
        print("4. EXIT")
        
        choice = input("ENTER YOUR CHOICE: ")
        
        if choice == '1':
            try:
                row_index = int(input("ENTER ROW NUMBER: "))
                result = detector.predict_outlier(X.iloc[row_index])
                print(f"\nRow {row_index} Outlier Detection Result:")
                print(f"Is Outlier: {result['is_outlier']}")
                print(f"Anomaly Score: {result['anomaly_score']:.4f}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            row_indices = input("ENTER MULTIPLE ROW SEPERATED BY COMMAS: ")
            indices = [int(idx.strip()) for idx in row_indices.split(',')]
            
            print("\nOUTLIER DETECTION RESULT:")
            for idx in indices:
                result = detector.predict_outlier(X.iloc[idx])
                print(f"Row {idx}: Is Outlier = {result['is_outlier']}")
        
        elif choice == '3':
            row_index = int(input("ENTER ROW NUMBER TO VISUALIZE: "))
            result = detector.visualize_row(row_index)
            print(f"\nOUTLIER STATUS: {result['is_outlier']}")
        
        elif choice == '4':
            print("EXIT HO GYA :) ")
            break
        
        else:
            print("INVALID CHOICE")

def main():
    interactive_outlier_detection()

if __name__ == '__main__':
    main()