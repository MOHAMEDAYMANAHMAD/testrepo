import pandas as pd
import numpy as np

class DataPrepKit:
    def __init__(self, file_path, file_format='csv'):
        self.df = self.data_reading(file_path, file_format)

    def data_reading(self, file_path, file_format='csv'):
        try:
            if file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_format.lower() == 'excel':
                df = pd.read_excel(file_path)
            elif file_format.lower() == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            return df
        except Exception as e:
            print(f"Error Data Reading : {str(e)}")
            return None

    def handling_missing_values(self, columns, strategy='remove'):
        try:
            df = self.df[columns]
            if strategy == 'remove':
                self.df[columns] = df.dropna()
            elif strategy == 'mean':
                self.df[columns] = df.fillna(df.mean())
            elif strategy == 'median':
                self.df[columns] = df.fillna(df.median())
        except Exception as e:
            print(f"Error Handling Missing Values: {str(e)}")  

    def data_summary(self, columns=None):
        try:
            summary = {}
            if columns is None:
                columns = self.df.columns
            elif isinstance(columns, str):
                columns = [columns]

            for column in columns:
                data = self.df[column]
                minimum = np.min(data)
                maximum = np.max(data)
                # Mean
                mean = np.mean(data)
                # Variance
                variance = np.var(data)
                # Standard Deviation
                std_dev = np.std(data)
                # Median
                median = np.median(data)
                # Quartiles
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)

                print(f'statistical summaries of {column} column :')
                print('\t Mean (Average)            :', mean)
                print('\t Variance                  :', variance)
                print('\t Standard Deviation        :', std_dev)
                print('\t Minimun                   :', minimum)
                print('\t 25th Percentile (Q1)      :', q1)
                print('\t Median (50th Percentile)  :', median)
                print('\t 75th Percentile (Q3)      :', q3)
                print('\t Maximun                   :', maximum)
                # Store results in dictionary
                summary[column] = {
                    'mean': mean,
                    'variance': variance,
                    'std_dev': std_dev,
                    'min': minimum,
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'max': maximum
                }
            return summary
        except Exception as e:
            print(f"Error Data Summary: {str(e)}")
            return None

    def categorical_data_encoding(self, columns):
        try:
            df = self.df.copy()
            if columns is None:
                columns = self.df.columns
            elif isinstance(columns, str):
                columns = [columns]

            for column in columns:
                try:
                    df[column] = df[column].astype('category').cat.codes
                except:
                    pass
            return df
        except Exception as e:
            print(f"Error Encoding Categorical Data: {str(e)}")
            return None


if __name__ == '__main__':
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eva', 'Frank', 'Grace', 'Henry', 'Isabel', 'Jack', 'Karen'],
        'Age': [25, 30, 22, 28, 35, 40, 26, 32, 29, 38, 27],
        'Gender': ['Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Salary': [50000, 60000, 55000, 70000, 75000, 80000, np.nan, 65000, 72000, 78000, 53000]
    }
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
    
    file_path = 'data.csv'
    data_prepper = DataPrepKit(file_path)    
    print(data_prepper.df)

    # Handling Missing Values
    data_prepper.handling_missing_values(columns=['Age', 'Salary'], strategy='mean')
    # Data Summary
    summary = data_prepper.data_summary('Salary')
    # Categorical Data Encoding
    encoding_df = data_prepper.categorical_data_encoding('Gender')

    print(data_prepper.df)
    print(encoding_df)