import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pycaret import regression,classification, clustering
        
 

# Function to load data
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
            st.text("DataFrame : ")
            st.write(data)
        elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
            st.text("DataFrame : ")
            st.write(data)
        else:
            st.error("Invalid file format. Please upload a CSV or Excel file.")
            return None
        return data

# Function to display basic statistics
def display_basic_stats(data):
    st.subheader("Basic Statistics")
    st.write(data.describe())

# Function to display data types
def display_data_types(data):
    st.subheader("Data Types")
    st.write(data.dtypes)

# Function to display missing values count
def display_missing_values(data,Data_type):
    #Separate Featurea as numerical and categorical 
     numeric_cols_df = data.select_dtypes(include=[np.number])
     categorical_cols_df = data.select_dtypes(include=[object])

     
     if Data_type == 'Num' :
         st.text("Count of Numerical Missing Values")
         st.write(numeric_cols_df.isnull().sum())
         st.text("DataFrame of Numerical Missing Values")
         st.write(numeric_cols_df[numeric_cols_df.isnull().any(axis=1)])
     elif Data_type == 'Cat' :
         st.text("Count of Categorical Missing Values")
         st.write(categorical_cols_df.isnull().sum())
         st.text("DataFrame of Categorical Missing Values")
         st.write(categorical_cols_df[categorical_cols_df.isnull().any(axis=1)])

    
# Function to drop missing values
def drop_missing_values(data,Data_type):
    #Separate Featurea as numerical and categorical 
     numeric_cols_df = data.select_dtypes(include=['int64', 'float64'])
     categorical_cols_df = data.select_dtypes(include=[object])

     
     if Data_type == 'Num' :
        st.write(numeric_cols_df.dropna(inplace=True))
        display_missing_values(numeric_cols_df,'Num')
     elif Data_type == 'Cat' :
         st.write(categorical_cols_df.dropna(inplace=True))
         display_missing_values(categorical_cols_df,'Cat')
     st.text("Missing Values Dropped successfully")
     
     
    
# Function to fill missing values
def fill_missing_values(data, filling_func):
    st.text("Missing Values filled sucessfully  ")
   #Separate Featurea as numerical and categorical 
    numeric_cols_df = data.select_dtypes(include=['int64', 'float64'])
    categorical_cols_df = data.select_dtypes(include=[object])
    
    # Numerical columns handling
    if filling_func == 'mean':  
        for col in numeric_cols_df.columns:      
            st.write(numeric_cols_df[col].fillna(numeric_cols_df[col].mean(), inplace=True)) 
        display_missing_values(numeric_cols_df,'Num')
    
    elif filling_func == 'median': 
        for col in numeric_cols_df.columns:         
            st.write(numeric_cols_df[col].fillna(numeric_cols_df[col].median(), inplace=True)) 
        display_missing_values(numeric_cols_df,'Num')
        
    elif filling_func == 'mode': 
         for col in categorical_cols_df.columns:         
             st.write(categorical_cols_df[col].fillna(categorical_cols_df[col].mode()[0], inplace=True)) 
         display_missing_values(categorical_cols_df,'Cat')
        
       
        
# Function to dispay duplicates in dataframe 
def display_duplicates(data):
    st.text("Duplicates in the data frame are :  ") 
    for col in data.columns:
        data[col] = data[col].duplicated()
    st.dataframe(data)

    
# Function to count duplicats in dataframe 
def count_duplicates(data):
    st.text("Sum of Duplicates in the data frame are :  ") 
    for col in data.columns:
        num_duplicates = data[col].duplicated().sum()
        st.write(col ,"  =  " ,num_duplicates )
 
 # Function to drop duplicats in dataframe 
def drop_duplicates(data):
     st.text(" Duplicates are dropped successfully ") 
     for col in data.columns:
         st.write(data.drop_duplicates(col, keep='last',inplace=True))
     count_duplicates(data)

    
# Function to display data visualizations using Matplotlib
def display_visualizations(data):
   
    st.text("Numerical columns are : ")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    st.write(numeric_cols)
    st.text("categorical columns are : ")
    categorical_cols = data.select_dtypes(include=[object]).columns
    st.write(categorical_cols)
    
    # Allow user to select column
    selected_column = st.selectbox("Select column to visualize:", data.columns)

   # Allow user to select plot type
    if data[selected_column].dtypes in ['int64', 'float64']:
        plot_type = st.selectbox("Select plot type:", ["Histogram"])
    else :
        st.write(data[selected_column].dtypes)
        plot_type = st.selectbox("Select plot type:", ["Bar Plot", "Pie chart"])
    # Histogram for numeric columns
    
    if plot_type == "Histogram":
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.hist(data[selected_column], bins=20, edgecolor='k')
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        
        
    elif plot_type == "Bar Plot":
        fig, ax = plt.subplots(figsize=(30, 15))
        value_counts = data[selected_column].value_counts()
        ax.bar(value_counts.index, value_counts.values, edgecolor='k')
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        ax.set_xticklabels(value_counts.index, rotation=45)
        st.pyplot(fig)
        
    elif plot_type == "Pie chart":
        
        fig, ax = plt.subplots(figsize=(30, 15))
        value_counts = data[selected_column].value_counts()
        ax.pie(value_counts,labels=value_counts.index.values.tolist(), autopct='%1.1f%%', startangle=90)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
 
    # Get a list of numerical column names
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create checkboxes for DataFrame columns
    selected_columns_to_scatter = st.multiselect('Select columns to visualize relation between:', numeric_cols )

    # Check if at least two columns were selected
    if len(selected_columns_to_scatter) == 2:
        # Create a scatter plot using the selected columns
        fig, ax = plt.subplots()
        ax.scatter(data[selected_columns_to_scatter[0]], data[selected_columns_to_scatter[1]])
        ax.set_xlabel(selected_columns_to_scatter[0])
        ax.set_ylabel(selected_columns_to_scatter[1])
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning('Please select two numerical columns for a scatter plot.')

# function to drop user selected columns
def drop_columns(data, selected_columns_to_drop) :
        data.drop(columns=selected_columns_to_drop,axis=1 , inplace= True)
        st.text("Data after columns dropped is ")
        st.write(data)
 


# Select best model         
def select_best_model(data,target,x):
    plotSelect =None 
    s = x.setup(data, target = target, session_id = 123)
    # compare all models
    best_model = x.compare_models()
    # check the best model
    st.write('Best Model is : ', best_model)

    if x == regression :
        # Plot model after Training
        plots = ["residuals", "error", "cooks", "rfe", "learning", "vc", "manifold", "feature", "feature_all", "parameter"]
        plotSelect = st.selectbox("Select Plot:",plots )
    elif x == classification :
        # Plot model after Training
        plots = ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary']
        plotSelect = st.selectbox("Select Plot:",plots )
    
    # functional API
    if plotSelect is not None:      
        if plotSelect:
            if best_model is not None :  
               # st.write('plot')
               try : 
                   fig2, ax2 = plt.subplots(figsize=(15, 15))
                   plot = x.plot_model(best_model, plot = plotSelect,display_format='streamlit')
                   st.write(plot)
               except :
                   st.warning('this plot is not available for that model')
                   
                 
def apply_clustering (dataset,model_name):
    s = clustering.setup(dataset, session_id = 123)
    model = clustering.create_model(model_name)
    # plot selected plot 
    cluster_plots =['cluster', 'tsne', 'elbow', 'silhouette', 'distance', 'distribution']
    plotSelect = st.selectbox("Select Plot:",cluster_plots )
    
    # functional API
    if plotSelect is not None:      
        if plotSelect:
            if model is not None :  
               # st.write('plot')
               try :           
                   plot = clustering.plot_model(model, plot = plotSelect,display_format='streamlit')
                   st.write(plot)
               except :
                   st.warning('this plot is not available for that model')
    
        
# Main function
def main():
    st.title("Automated EDA App")
    st.write("Upload a CSV or Excel file for Exploratory Data Analysis.")

    # File upload
    file = st.file_uploader("Upload a file", type=["csv", "xls", "xlsx"])

    if file is not None:
        data = load_data(file)
        if data is not None:
            display_basic_stats(data)
            display_data_types(data) 
     

    # Missong values Operations                  
            st.subheader("Missing Values Operations")
            
            # Create a checkbox to choose if operations will de done on specfic column or all dataset
            selected_item = st.radio('Select data to which functions is applied:', ['Specific Columns', 'Whole dataset'],key="Rs_1")
            # Check if the checkbox is selected
            if selected_item == 'Specific Columns' : 
                
                # Create checkboxes for DataFrame columns
                selected_columns = st.multiselect('Select columns to apply operation on:', data.columns)
                    # Display selected columns
                if selected_columns:
                    data = data[selected_columns]
                else:
                    st.warning('Please select at least one column.')
                                    
            elif  selected_item == 'Whole dataset':
                data = data
            
            else :
             st.warning('Please select which data we need to apply on')
            
            
            functions = ["Default","Count Missing Values", "Fill Missing Values with mean","Fill Missing Values with median", "Drop Missing Values"]
            selected_function = st.selectbox("Select function to apply on Numerical Features: ", functions)
            
            if selected_function == "Count Missing Values":
                display_missing_values(data,'Num')
            elif selected_function == "Fill Missing Values with mean":
                fill_missing_values(data,'mean')
            elif selected_function == "Fill Missing Values with median":
                fill_missing_values(data,'median')
            elif selected_function == "Drop Missing Values":
                drop_missing_values(data,'Num')
                

            functions = ["Skip","Count Missing Values", "Fill Missing Values with mode", "Drop Missing Values"]
            selected_function = st.selectbox("Select function to apply on Categorical Features: ", functions)
            
            if selected_function == "Count Missing Values":
                display_missing_values(data,'Cat')
            elif selected_function == "Fill Missing Values with mode":
                fill_missing_values(data,'mode')
            elif selected_function == "Drop Missing Values":
                drop_missing_values(data,'Cat')
                               


    # Duplicated values Operations            
            st.subheader("Duplicated Data Operations")
            functions = ["Skip", "Display Duplicated Values", "Count Duplicates in each column", "Drop Duplicates"]
            selected_function = st.selectbox("Select function to apply:", functions)


            if selected_function == "Display Duplicated Values":
                 display_duplicates(data)
            elif selected_function == "Count Duplicates in each column":
                count_duplicates(data)
            elif selected_function == "Drop Duplicates":
                drop_duplicates(data)
             
      #Data Visualiztion
            st.subheader("Data Visualiztion")
            display_visualizations(data) 
            
      #Drop Selected columns 
            st.subheader("Drop Column Operation")
            # Create checkboxes for DataFrame columns
            selected_columns_to_drop = st.multiselect('Select columns to Drop:', data.columns,key="ms_1")
            if len(selected_columns_to_drop) >0:
                drop_columns(data , selected_columns_to_drop);
 
        # Model Selection and columns to operate on 
        
        # Initialize dataset and target variables
            st.subheader("Model Selection")
            catg = ['Clustering' , 'others' ]
            Task_Cat = st.selectbox('Select Task Category : ' , catg ,key = "ms_5")
            if Task_Cat == 'Clustering' : 
                clusr_models= [ 'kmeans' , 'ap', 	'meanshift' , 'sc' ,  'hclust' , 'dbscan' , 'optics' ,  'birch' , 'kmodes' ]
                cluster_model = st.selectbox('Select Target needed :', clusr_models ,key="ms_4")
                st.write('You Selected clustering Task')
                apply_clustering(data,cluster_model)
                
            elif Task_Cat == 'others' : 
                                         
                selected_data = st.radio('Select data to which functions is applied:', ['Specific Columns', 'Whole dataset'],key="Rs_2")
                # Check if the checkbox is selected
                dataset = None
                target = None
             
                if selected_data == 'Specific Columns' : 
                    
                    # Create checkboxes for DataFrame columns
                    target_column = st.selectbox('Select Target needed :', data.columns,key="ms_2")
                    selected_columns_to_predict_on = st.multiselect('Select columns to apply operation on:', data.columns,key="ms_3")
                        # Display selected columns
                    if selected_columns_to_predict_on and target_column :
                        dataset = data[selected_columns_to_predict_on]
                        target = data[target_column]
                        target_data_type = target.dtypes
                     #   st.write('DataFrame is',dataset);
                     #   st.write('Target is',target);
                    else:
                        st.warning('Please select at least one column.')
                                        
                elif  selected_data == 'Whole dataset':
                    
                    dataset = data.iloc[:,:-1]
                    target= data.iloc[:,-1]
                    target_data_type = target.dtypes
                  #  st.write('DataFrame is',dataset);
                  #  st.write('Target is',target);
                
                else :
                 st.warning('Please select which data we need to apply on')
                 
                
                if dataset is None :   
                    st.warning('Please Select columns to apply predict or select whole dataset')         
                else :
                    st.write('dataset  ' , dataset) 
                    # Create a checkbox to choose if operations will de done on specfic column or all dataset
                    selected_task_mode = st.radio('Select task  mode Automatic or Manual Mode for task category:', ['Automatic', 'Manual'],key="Rs_3")
                    if selected_task_mode == 'Automatic' :          
                        is_numeric = pd.api.types.is_numeric_dtype(target_data_type)
                        if is_numeric :
                            target_min = target.min()
                            target_max = target.max()
                            target_sum = target_min + target_max
                            if target_sum > 10 :
                                x = regression 
                                st.write('it will be a regression problem')
                            else :
                                x=classification
                                st.write('it will be a classification problem')
                            select_best_model(dataset,target,x)
                        else :
                            x=classification
                            st.write('classification')
                            select_best_model(dataset,target,x)
                            
                    elif selected_task_mode == 'Manual': 
                        selected_task = st.radio('Select task category:', ['Regression', 'Classification'],key="Rs_4")
                        if selected_task == 'Regression':
                            x=regression
                            st.write('You Selected Regression Task')
                            select_best_model(dataset,target,x)
                            
                        elif selected_task == 'Classification':
                            x=classification
                            st.write('You Selected Classification Taskk')
                            select_best_model(dataset,target,x)
    
                    
    
                        else :
                            st.warning('Please Select at least one task Regression or Classification ')
                            
                    else :
                        st.warning('Please Select at least one task Mode ')
                        
            else :
                st.warning('Please Select problem category ')
                    
                                         

if __name__ == "__main__":
    main()