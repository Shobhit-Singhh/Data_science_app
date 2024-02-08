import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
import scipy.cluster.hierarchy as sch
import io
import os
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import pygwalker as pyg
import streamlit.components.v1 as components
w.filterwarnings("ignore")


def compare_distribution(df, dummy, col=None):
    if col is None:
        col = df.columns
    for i in col:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Distribution for {i}'])
        # Add histograms for the current column to the corresponding subplot
        fig.add_trace(go.Histogram(x=df[i], nbinsx=50, name='Before', marker=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Histogram(x=dummy[i], nbinsx=50, name='After', marker=dict(color='red', opacity=0.5)), row=1, col=1)
        fig.update_layout(title_text=f'Distribution Comparison for {i}', showlegend=False)
        st.plotly_chart(fig)
def compare_covariance(df, dummy, col,lg=True):
    num_cols = df.select_dtypes(include=['number']).columns
    
    for i in col:
        if i in num_cols:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Before {i}', f'After {i}'])
            # Add histograms for the current column to the corresponding subplot
            fig.add_trace(go.Histogram(x=df[i], nbinsx=50, name='Before', marker=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Histogram(x=dummy[i], nbinsx=50, name='After', marker=dict(color='red', opacity=0.5)), row=1, col=1)
            fig.update_layout(title_text=f'Covariance Comparison for {i}', showlegend=lg)
            st.plotly_chart(fig)
            num_df = df[num_cols]
            num_dummy = dummy[num_cols]
            st.write(num_df.corr()[i].to_frame().T)
            st.write(num_dummy.corr()[i].to_frame().T)
        else:
            st.warning(f"{i} is not a numerical column")
def plot_categorical_distribution(df, dummy, columns, lg=True):
    num_columns = len(columns)
    fig = make_subplots(rows=1, cols=1, subplot_titles=sum([[f'Before {col}', f'After {col}'] for col in columns], []))

    for i, col in enumerate(columns, start=1):
        # Plotting the distribution before imputation
        before_trace = px.histogram(df, x=col, title=f'Distribution of {col}').data[0]
        before_trace.marker.color = 'red'
        fig.add_trace(before_trace, row=1, col=1)

        after_trace = px.histogram(dummy, x=col, title=f'Distribution of {col}').data[0]
        after_trace.marker.color = 'green'
        fig.add_trace(after_trace, row=1, col= 1)

        fig.update_layout(title_text=f'Distribution Comparison for {", ".join(columns)}', showlegend=lg)
        fig.update_xaxes(type='category')
        fig.update_layout(title_text=f'distribution Comparison for {col}', showlegend=lg)
        st.plotly_chart(fig)
def uplode_and_reset():
    with tab1:
        st.header("Upload dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        reset_button = st.button("Reset dataset")
        
        if st.session_state.get('df') is not None:
            df = st.session_state.df
        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        else:
            df = pd.read_csv(os.path.join('data','Titanic.csv'))
            st.session_state.df = df
            
        if reset_button:
            st.session_state.df = None
            st.success("Successfully reset the dataset.")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(os.path.join('data','Titanic.csv'))
        st.write("To make the uploaded file visible")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                padding: 2 px;
                text-align: center;
                font-size: 14px;
                color: #555;
            }
        </style>
        <div class="footer">
            Data Science App by Shobhit Singh, 
            <a class="linkedin" href="https://www.linkedin.com/in/shobhit-singhh/" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
        return df
def sidebar(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.download_button(label="Download CSV file",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name='output.csv',
                        key='download_button')
    


def show_dataset_shape(df):
    st.write(f"The shape of the dataframe = {df.shape}")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_sample_dataset(df):
    st.header("Sample of the dataset")
    sample_size = st.slider("Enter the sample size of data ", 1, 100, 10)
    rows = st.radio("Select the rows", ("Top rows", "Bottom rows", "Random rows"))
    if rows == "Top rows":
        st.write(df.head(sample_size))
    elif rows == "Bottom rows":
        st.write(df.tail(sample_size))
    else:
        st.write(df.sample(sample_size))
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_dataset_description(df):
    st.header("Dataset description")
    st.write(df.describe().T)
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_null_values(df):
    st.header("Null values")
    st.write(pd.DataFrame(df.isnull().sum(), columns=['Null Count']).T)
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_duplicate_values(df):
    st.header("Duplicate values")
    st.write(f"Total duplicate rows in the dataset = {df.duplicated().sum()}")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_correlation_metric(df):
    st.header("Correlation metric")
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = st.multiselect("Select columns for correlation", numeric_columns)
    if selected_columns:
        st.write("Correlation Matrix:")
        st.write(df[selected_columns].corr())
    else:
        st.write("No columns selected for correlation.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def know_data(df):
    with tab2:
        expander_1 = st.expander("Know your data")
        expander_1.header("Know your data")
        expander_1.write("The first step in any data science project is to understand the data. This section provides a brief overview of the dataset and its features.")
        expander_1.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        show_shape = expander_1.checkbox("Show dataset shape")
        show_sample = expander_1.checkbox("Show sample of the dataset")
        show_describe = expander_1.checkbox("Show dataset description")
        show_null = expander_1.checkbox("Show null values")
        show_duplicate = expander_1.checkbox("Show duplicate values")
        show_corr = expander_1.checkbox("Show correlation metric")
        
    if show_shape:
        show_dataset_shape(df)
    if show_sample:
        show_sample_dataset(df)
    if show_describe:
        show_dataset_description(df)
    if show_null:
        show_null_values(df)
    if show_duplicate:
        show_duplicate_values(df)
    if show_corr:
        show_correlation_metric(df)


def remove_duplicate_rows(df):
    try:
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed successfully.")
    except Exception as e:
        st.error(f"Error removing duplicate rows: {str(e)}")
    st.session_state.df = df
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def remove_columns(df):
    st.header("Remove columns")
    cols = st.multiselect("Select columns to remove", df.columns)
    remove_button = st.button("Remove Columns")
    if remove_button:
        try:
            df.drop(cols, axis=1, inplace=True)
            st.success(f"{cols} Columns removed successfully.")
        except Exception as e:
            st.error(f"Error removing columns: {str(e)}")
    st.session_state.df = df
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def rename_columns(df):
    st.header("Rename columns")
    st.write("Enter new column names:")
    selected_column = st.selectbox("Select a column for rename:", df.columns)
    new_name = st.text_input("Enter new column name:")
    
    rename_button = st.button("Rename Column")
    if rename_button:
        try:
            df.rename(columns={selected_column: new_name}, inplace=True)
            st.success(f"Successfully renamed {selected_column} to {new_name}.")
        except Exception as e:
            st.error(f"Error renaming {selected_column} to {new_name}: {str(e)}")
    st.session_state.df = df
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def convert_data_types(df):
    st.header("Convert data types")
    selected_column = st.selectbox("Select a column for conversion:", df.columns)
    new_dtype = st.selectbox("Select new data type:", ["object", "int64", "float64","bool"])
    convert_button = st.button("Convert Data Type")
    if convert_button:
        try:
            df[selected_column] = df[selected_column].astype(new_dtype)
            st.success(f"Successfully converted {selected_column} to {new_dtype}.")
        except Exception as e:
            st.error(f"Error converting {selected_column} to {new_dtype}: {str(e)}")
    st.session_state.df = df
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def add_bucket_column(df):
    st.header("Add bucket column")
    selected_column = st.selectbox("Select a column to add the bucket for:", df.columns)
    new_column_name = st.text_input("Enter new column name:", f"{selected_column}_bucket")
    bucket_type = st.selectbox("Select bucket type:", ["Select the number of bins", "Select the bin intervals"])
    bucket_size = st.slider("Select bucket size:", 1, 100, 10)
    bucket_button = st.button("Add Bucket Column")
    if bucket_button:
        try:
            if bucket_type == "Select the number of bins":
                df[new_column_name] = pd.qcut(df[selected_column], q=bucket_size)
            elif bucket_type == "Select the bin intervals":
                df[new_column_name] = pd.cut(df[selected_column], bins=bucket_size)
            st.success(f"Successfully added {new_column_name} as a bucket column.")
        except Exception as e:
            st.error(f"Error adding {new_column_name} as a bucket column: {str(e)}")
    st.session_state.df = df
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def clean_data(df):
    with tab2:
        expander_2 = st.expander("Clean your data")
        with expander_2:
            st.header("Clean your data")
            st.write("This section allows you to clean the dataset by removing null values and duplicate rows.")
            st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        

    if expander_2.checkbox("Remove duplicate rows"): remove_duplicate_rows(df)
    if expander_2.checkbox("Rename columns"): rename_columns(df)
    if expander_2.checkbox("Convert data types"): convert_data_types(df)
    if expander_2.checkbox("Add bucket column"): add_bucket_column(df)


def plot_with_pygwalker(df):
    pyg_html = pyg.to_html(df)
    components.html(pyg_html, height=900, scrolling=True)
def split_dataset(df):
    unique_value = st.slider("Enter the number of unique values for threshold", 1, 20, 5)
    continous_columns = []
    categorical_columns = []
    discrete_columns = []

    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > unique_value:
            discrete_columns.append(col)
            
        elif df[col].nunique() <= unique_value:
            categorical_columns.append(col)
            
        else:
            continous_columns.append(col)
            

    col1, col2, col3 = st.columns(3)

    col1.write("Continuous Columns:")
    col1.write(continous_columns)


    col2.write("Categorical Columns:")
    col2.write(categorical_columns)


    col3.write("Discrete Columns:")
    col3.write(discrete_columns)
    
    return continous_columns, categorical_columns, discrete_columns
def Scatter_plot(df, continous_columns, categorical_columns, discrete_columns, show_legend=True):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns) + list(continous_columns))
    y_column = st.selectbox("Y-axis", [None] + list(categorical_columns) + list(continous_columns))
    hue_column = st.selectbox("Hue", [None] + list(categorical_columns))
    size_column = st.selectbox("Size", [None] + list(categorical_columns))
    style_column = st.selectbox("Style", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    sizes_tuple = st.slider("Select Sizes Range", min_value=0, max_value=200, value=(50, 200))
    fig, ax = plt.subplots()

    try:
        sns.scatterplot(x=x_column, y=y_column, data=df, hue=hue_column, size=size_column, style=style_column, palette=palette, sizes=sizes_tuple, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)  # Turn off legend
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting scatter plot: {str(e)}")
def Bar_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.barplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting bar plot: {str(e)}")
def Box_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting box plot: {str(e)}")
def Histogram(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    bins = st.slider("Select Number of Bins", min_value=1, max_value=100, value=30)
    color = st.color_picker("Select Color", value="#1f77b4")

    fig, ax = plt.subplots()
    try:
        sns.histplot(df[x_column], bins=bins, color=color, kde=False)
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {x_column}")
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting histogram: {str(e)}")
def Heatmap(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    cmap = st.selectbox("Select Colormap", ("viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"))
    fig, ax = plt.subplots()
    numeric_columns = df.select_dtypes(include=['number']).columns
    try:
        sns.heatmap(df[numeric_columns].corr(), annot=show_legend, cmap=cmap,fmt=".1f")
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting heatmap: {str(e)}")
def Count_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.countplot(x=x_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting count plot: {str(e)}")
def Pie_plot(df, continous_columns, categorical_columns, discrete_columns, show_legend=True):
    st.header("Select Plot Options")
    column = st.selectbox("Column", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        counts = df[column].value_counts()
        counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colormap=palette, ax=ax)
        plt.title(f"Pie Plot of {column}")
        
        if not show_legend:
            ax.legend().set_visible(False)  # Turn off legend
            
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting pie plot: {str(e)}")
def Distplot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    color = st.color_picker("Select Color", value="#1f77b4")
    bins_slider = st.slider("Select Number of Bins", min_value=1, max_value=100)

    fig, ax = plt.subplots()
    try:
        sns.distplot(df[x_column], bins=bins_slider, color=color,hist_kws=dict(edgecolor="k", linewidth=.5))
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution Plot of {x_column}")
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting distplot: {str(e)}")
def Line_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    st.header("Select Plot Options")
    x_column = st.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.lineplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting line plot: {str(e)}")
def plot_data(df):
    with tab2:
        extender_3 = st.expander("Plot your data")
        with extender_3:
            plot = st.checkbox("plot your data")
            
    if plot:
        plot_with_pygwalker(df)
            
    # extender_3.header("Plot your data")
    # graph = extender_3.selectbox("Select Graph", ("None","Scatter Plot", "Bar Plot", "Box Plot", "Histogram", "Heatmap", "Count Plot", "Pie Plot", "Distplot", "Line Plot"))
    # split_data = extender_3.checkbox("Split data into continous and categorical columns")
    # show_legend = extender_3.checkbox("Show Legend", value=True)
    # continous_columns, categorical_columns, discrete_columns = np.array(df.columns), np.array(df.columns), np.array(df.columns)
    
    # if split_data:    
    #     continous_columns, categorical_columns, discrete_columns = split_dataset(df)
    # if graph == "None":
    #     pass
    # elif graph == "Scatter Plot":
    #     Scatter_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Bar Plot":
    #     Bar_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Box Plot":
    #     Box_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Histogram":
    #     Histogram(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Heatmap":
    #     Heatmap(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Count Plot":
    #     Count_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Pie Plot":
    #     Pie_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Distplot":
    #     Distplot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    # elif graph == "Line Plot":
    #     Line_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)


def remove_missing_values(df):
    try:
        st.header("Cautions: Complete Case Analysis")
        st.write("Complete case analysis (CCA), also called listwise deletion of cases, consists in discarding observations where values in any of the variables are missing. Result loss of information.")
        st.write("Assuming that the data are missing completely at random (MCAR), the complete case analysis is unbiased. However, if data are missing at random (MAR) or not at random (MNAR), then complete case analysis leads to biased results.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
        tab_rmv_miss1,tab_rmv_miss2=st.tabs(["Remove Colums","Remove Rows"])
        with tab_rmv_miss1:
            col = st.multiselect("Select the column", df.columns)
            if st.checkbox("Remove Columns") and col:
                df.drop(col, axis=1, inplace=True)
                st.session_state.df = df
                st.success(f"{col} column(s) removed successfully.")

        with tab_rmv_miss2:
            percent_missing_lowerlimit, percent_missing_upperlimit  = st.slider("Select the range of missing values: Recommended (0-5) ", 0, 100, (0, 5))
            col_missing_percent = [var for var in df.columns if df[var].isnull().mean()*100 > percent_missing_lowerlimit and df[var].isnull().mean()*100 < percent_missing_upperlimit]
            col = st.multiselect("Select the column", col_missing_percent)
            dummy = df.dropna(subset=col)
            compare_distribution(df,dummy)
            
            if st.checkbox("Confirm the removal of rows with missing values"):
                df.dropna(subset=col, inplace=True)
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
                st.success(f"{col} column(s) removed successfully.")

    except Exception as e:
        st.error(f"Error removing missing values: {str(e)}")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def missing_values_imputation(df):
    st.header("Missing Values Imputation")
    st.write("Missing values imputation is the process of replacing missing data with substituted values. This section allows you to impute missing values in the dataset.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
    lg = st.checkbox("Show Legend")
    st.header("For Numerical Columns:")
    numerical_cols_withNA = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()]
    col = st.multiselect("Select the numerical column", numerical_cols_withNA)

    tab_num_imput1,tab_num_imput2,tab_num_imput3,tab_num_imput4,tab_num_imput5,tab_num_imput6=st.tabs(["Mean","Median","Mode","Random","End of Distribution","KNN"])

    with tab_num_imput1:
        mean_dummy = df.copy()
        st.warning("Warning: Mean imputation is sensitive to outliers, coversion and distribution.")
        for c in col:
            mean_dummy[c].fillna(df[c].mean(), inplace=True)
        compare_covariance(df,mean_dummy,col,lg)
        if st.checkbox("Confirm the mean imputation"):
            df=mean_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_num_imput2:
        median_dummy = df.copy()
        st.warning("Warning: Median imputation is sensitive to outliers, coversion and distribution.")
        for c in col:
            median_dummy[c].fillna(df[c].median(), inplace=True)
        compare_covariance(df,median_dummy,col,lg)
        if st.checkbox("Confirm the median imputation"):
            df=median_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_num_imput3:
        mode_dummy = df.copy()
        st.warning("Warning: Mode imputation is sensitive to outliers, coversion and distribution.")
        for c in col:
            mode_dummy[c].fillna(df[c].mode().mean(), inplace=True)
        compare_covariance(df,mode_dummy,col,lg)
        if st.checkbox("Confirm the mode imputation"):
            df=mode_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_num_imput4:
        random_dummy = df.copy()
        for c in col:
            random_sample = df[c].dropna().sample(df[c].isnull().sum())
            random_sample.index = df[df[c].isnull()].index
            random_dummy.loc[df[c].isnull(), c] = random_sample
        compare_covariance(df, random_dummy, col,lg)
        if st.checkbox("Confirm the random imputation"):
            df=random_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_num_imput5:
        st.write("End of Distribution")
        operation = st.selectbox("Select the column", ['None','-1','max','min','0','max + 3*sd','min - 3*sd'])
        summery_5 = pd.DataFrame([df[col].min(),df[col].quantile(0.25),df[col].quantile(0.5),df[col].quantile(0.75),df[col].max()],index=['min','Q1','Q2','Q3','max']).T
        st.write(summery_5)
        eod_dummy = df.copy()
        for c in col:
            if operation == '-1':
                eod_dummy[c].fillna(-1, inplace=True)
            elif operation == 'max':
                eod_dummy[c].fillna(df[c].max(), inplace=True)
            elif operation == 'min':
                eod_dummy[c].fillna(df[c].min(), inplace=True)
            elif operation == '0':
                eod_dummy[c].fillna(0, inplace=True)
            elif operation == 'max + 3*sd':
                # Set values greater than max + 3*sd to max + 3*sd
                threshold = eod_dummy[c].max() + 3 * eod_dummy[c].std()
                eod_dummy[c].fillna(threshold, inplace=True)
            elif operation == 'min - 3*sd':
                # Set values less than min - 3*sd to min - 3*sd
                threshold = eod_dummy[c].min() - 3 * eod_dummy[c].std()
                eod_dummy[c].fillna(threshold, inplace=True)
        compare_covariance(df,eod_dummy,col,lg)
        if st.checkbox("Confirm the end of distribution imputation"):
            df=eod_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_num_imput6:
        st.write("KNN")
        neighbour = st.slider("Select the number of neighbors", 1, 10, 3)
        knn_dummy = df.copy()
        for c in col:
            imputer = KNNImputer(n_neighbors=neighbour)
            knn_dummy[c] = imputer.fit_transform(df[[c]])[:, 0]
        compare_covariance(df,knn_dummy,col,lg)
        if st.checkbox("Confirm the KNN imputation"):
            df=knn_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    st.header("For Categorical Columns:")
    
    cat_cols_withNA = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()]
    col_cat = st.multiselect("Select the catagorical column", cat_cols_withNA)

    tab_cat_imput1, tab_cat_imput2 = st.tabs(["Mode", "Replace with 'Missing value' tag"])

    with tab_cat_imput1:
        mode_cat_dummy = df.copy()
        st.warning("Warning: Mode imputation is sensitive to outliers, coversion, and distribution for categorical columns.")
        for c in col_cat:
            mode_cat_dummy[c].fillna(df[c].mode(), inplace=True)
        plot_categorical_distribution(df, mode_cat_dummy, col_cat,lg)
        if st.checkbox("Confirm the Categorical Mode imputation"):
            df=mode_cat_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

    with tab_cat_imput2:
        missing_value_tag_cat_dummy = df.copy()
        for c in col_cat:
            missing_value_tag_cat_dummy[c].fillna("Missing value", inplace=True)
        plot_categorical_distribution(df, missing_value_tag_cat_dummy, col_cat,lg)
        if st.checkbox("Confirm the 'Missing' tag imputation"):
            df=missing_value_tag_cat_dummy
            st.session_state.df = df
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def outliers_detection(df):
    st.header("Outliers Detection")
    st.write("An outlier is a data point that differs significantly from other observations. This section allows you to detect outliers in the dataset.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    col = st.multiselect("Select the column ", df.select_dtypes(include=['number']).columns)
    if col:
        tab_Zscore, tab_IQR, tab_percentile = st.tabs(["Z score", "IQR", "Percentile"])
        
        with tab_Zscore:
            st.write("Z score")
            threshold = st.slider("Select the threshold Z score", 0.0, 3.0, 1.5, step=0.01)
            z_score_dummy = df.copy()
            for c in col:
                z_scores = (z_score_dummy[c] - z_score_dummy[c].mean()) / z_score_dummy[c].std()
                z_score_dummy[c] = np.where(np.abs(z_scores) > threshold, np.nan, z_score_dummy[c])
            compare_distribution(df, z_score_dummy, col)
            if st.checkbox("Confirm the Z-score outliers detection"):
                df = z_score_dummy
                st.session_state.df = df
    
        with tab_IQR:
            st.write("IQR")
            threshold = st.slider("Select the threshold for IQR multiplication", 0.0, 3.0, 1.5, step=0.01)
            IQR_dummy = df.copy()
            IQR=IQR_dummy[c].quantile(0.75) - IQR_dummy[c].quantile(0.25)
            for c in col:
                lower_limit = IQR_dummy[c].quantile(0.25) - threshold * IQR
                upper_limit = IQR_dummy[c].quantile(0.75) + threshold * IQR
                IQR_dummy[c] = np.where((IQR_dummy[c] < lower_limit) | (IQR_dummy[c] > upper_limit), np.nan, IQR_dummy[c])
            compare_distribution(df, IQR_dummy,col)
            if st.checkbox("Confirm the IQR outliers detection"):
                df = IQR_dummy
                st.session_state.df = df

        with tab_percentile:
            st.write("Percentile")
            lower_percentile, upper_percentile = st.slider("Select the range of percentile", 0, 100, (0, 5))
            percentile_dummy = df.copy()
            for c in df.select_dtypes(include=['number']).columns:
                lower_limit = percentile_dummy[c].quantile(lower_percentile / 100)
                upper_limit = percentile_dummy[c].quantile(upper_percentile / 100)
                percentile_dummy[c] = np.where((percentile_dummy[c] < lower_limit) | (percentile_dummy[c] > upper_limit), np.nan, percentile_dummy[c])
            compare_distribution(df, percentile_dummy,col)
            if st.checkbox("Confirm the percentile outliers detection"):
                df = percentile_dummy
                st.session_state.df = df

    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def feature_encoding(df):

    st.header("Feature Encoding")
    st.write("Feature encoding is the technique of converting categorical data into numerical data. This section allows you to encode features in the dataset.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    cardinality_threshold = st.slider("Cardinality: Enter the number of unique values for threshold", 1, 20, 5)
    col = df.columns[df.apply(lambda x: x.nunique()) < cardinality_threshold]

    st.write([col])
    tab_onehot, tab_ordinal, tab_frequency, tab_target = st.tabs(["One Hot", "Ordinal", "Frequency", "Target"])

    with tab_onehot:
        st.write("One-hot encoding is preferred when dealing with categorical variables with no inherent order, as it avoids introducing unintended ordinal relationships and allows machine learning models to treat each category independently, preserving the nominal nature of the data.")
        target_col = st.multiselect("Select the column for One-hot Encoding", col)

        if st.checkbox("Show One-hot encoding") and target_col:
            one_hot_dummy = df.copy()
            one_hot_dummy = pd.get_dummies(one_hot_dummy, columns=target_col, drop_first=True)

            # Filter columns that start with the selected target_col
            selected_columns = [col for col in one_hot_dummy.columns if any(col.startswith(prefix) for prefix in target_col)]

            unique_values_df = pd.DataFrame({
                "Column": selected_columns,
                "Unique_Values_after_One_Hot_Encoding": [one_hot_dummy[col].unique() for col in selected_columns]
            })

            # Convert the 'Unique_Values_after_One_Hot_Encoding' column values to strings
            unique_values_df['Unique_Values_after_One_Hot_Encoding'] = unique_values_df['Unique_Values_after_One_Hot_Encoding'].astype(str)

            st.write(unique_values_df)

            if st.checkbox("Confirm the One Hot encoding"):
                df = one_hot_dummy
                st.session_state.df = df
    with tab_ordinal:
        st.write("Ordinal encoding is preferred when dealing with categorical variables with an inherent order, as it introduces ordinal relationships and allows machine learning models to treat each category independently, preserving the ordinal nature of the data.")
        target_col = st.selectbox("Select the column for Ordinal Encoding", col)

        unique_values = df[target_col].unique().tolist()
        order = st.multiselect("Select the order of unique values", unique_values)

        if  st.checkbox("Show Ordinal encoding") and order:
            ordinal_dummy = df.copy()
            oe = OrdinalEncoder(categories=[order])
            ordinal_dummy[target_col+'_ordinal'] = oe.fit_transform(ordinal_dummy[[target_col]]).astype(int)

            unique_values_df = pd.DataFrame({
                "Column": [col for col in ordinal_dummy.columns if col.startswith(target_col)],
                "Unique_Values_after_Ordinal_Encoding": [ordinal_dummy[col].unique() for col in ordinal_dummy.columns if col.startswith(target_col)]
            })

            # Convert the 'Unique_Values_after_Ordinal_Encoding' column values to strings
            unique_values_df['Unique_Values_after_Ordinal_Encoding'] = unique_values_df['Unique_Values_after_Ordinal_Encoding'].astype(str)

            st.write(unique_values_df)

            if st.checkbox("Confirm the Ordinal encoding"):
                df = ordinal_dummy
                st.session_state.df = df
        else:
            st.warning("Please select the order of unique values for ordinal encoding.")
    with tab_frequency:
        st.write("Frequency encoding is the technique of converting categorical data into their respective frequency count offering an informative numerical representation, particularly useful for high-cardinality features.")
        target_col = st.multiselect("Select the column for Frequency Encoding", df.columns)

        if st.checkbox("Show Frequency encoding") and target_col:
            frequency_dummy = df.copy()  # Copy the original DataFrame outside the loop

            for c in target_col:
                frequency_encoding = frequency_dummy[c].value_counts().to_dict()
                frequency_dummy[c+'_freq'] = frequency_dummy[c].map(frequency_encoding)

            unique_values_df = pd.DataFrame({
                "Column": [col for col in frequency_dummy.columns if any(col.startswith(prefix) for prefix in target_col)],
                "Unique_Values_after_Frequency_Encoding": [frequency_dummy[col].unique() for col in frequency_dummy.columns if any(col.startswith(prefix) for prefix in target_col)]
            })

            # Convert the 'Unique_Values_after_Frequency_Encoding' column values to strings
            unique_values_df['Unique_Values_after_Frequency_Encoding'] = unique_values_df['Unique_Values_after_Frequency_Encoding'].astype(str)

            st.write(unique_values_df)

            if st.checkbox("Confirm the Frequency encoding"):
                df = frequency_dummy
                st.session_state.df = df
    with tab_target:
        st.write("Target encoding is a technique where each category of unique value is replaced with the mean of the target variable for that category.")
        target_col = st.multiselect("Select the column for Mean Target Encoding", df.columns)
        target = st.selectbox("Select the target column", df.columns)

        if st.checkbox("Show Mean Target encoding") and target:
            target_dummy = df.copy()

            for col in target_col:
                target_map = target_dummy.groupby(col)[target].mean().to_dict()
                target_dummy[col + '_target'] = target_dummy[col].map(target_map)

            unique_values_df = pd.DataFrame({
                "Column": [col + '_target' for col in target_col],
                "Unique_Values_after_Target_Encoding": [target_dummy[col + '_target'].unique() for col in target_col]
            })

            # Convert the 'Unique_Values_after_Target_Encoding' column values to strings
            unique_values_df['Unique_Values_after_Target_Encoding'] = unique_values_df['Unique_Values_after_Target_Encoding'].astype(str)

            st.write(unique_values_df)

            if st.checkbox("Confirm the Target encoding"):
                df = target_dummy
                st.session_state.df = df
def feature_scaling_transformation(df):
    st.header("Feature Scaling")
    st.write("Feature scaling is a method used to normalize the range of independent variables or features of data. This section allows you to scale features in the dataset.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    
    extender_describe_scaling = st.expander("Describe the all scaling techniques")
    with extender_describe_scaling:
        st.write("Standardization: Standardization is suitable when the features have different units or scales, and it's important to center them around the mean with a unit standard deviation. Use it when the data has a normal distribution and outliers are not a concern.")
        st.latex(r'x_{new} = \frac{x - \mu}{\sigma}')
        st.write("MinMax Scaler: MinMax Scaler is useful when you want to scale features to a specific range. However, it may not be suitable for datasets with outliers, as it is sensitive to extreme values.")
        st.latex(r'x_{new} = \frac{x - min(x)}{max(x) - min(x)}')
        st.write("MaxAbs Scaler: MaxAbs Scaler is suitable when you want to scale features to the [-1, 1] range without shifting the data. Use it when preserving the sign of the data is essential, and the data has outliers.")
        st.latex(r'x_{new} = \frac{x}{max(abs(x))}')
        st.write("Robust Scaler: Robust Scaler is a good choice when dealing with datasets containing outliers. It scales data based on the interquartile range (IQR) and is less affected by outliers. Use it when the data has outliers or is not normally distributed.")
        st.latex(r'x_{new} = \frac{x - Q_1(x)}{Q_3(x) - Q_1(x)}')
        st.write("Quantile Transformer Scaler: Use this scaler when you need to transform features to follow a uniform or normal distribution. It's robust against outliers and can be beneficial when working with non-normally distributed data.")
        st.latex(r'x_{new} = F^{-1}(x)')
        st.write("Box-Cox Transformation: Box-Cox Transformation is a combined version of log and square transform, suitable for stabilizing variance and making the data more normally distributed. Use it when the data is positive and right-skewed. If value is zero, then replace it with very small arbitrary value to avoid error.")
        st.latex(r'x_{new} = \begin{cases} \frac{x^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln{(x)} & \text{if } \lambda = 0 \end{cases}')
        st.write("Yeo-Johnson Transformation: Yeo-Johnson Transformation is a modified version of Box-Cox that works with positive and negative values. It's suitable for stabilizing variance and making the data more normally distributed. Use it when the data contains both positive and negative values.")
        st.latex(r'x_{new} = \begin{cases} ((x+1)^{\lambda} - 1)/\lambda & \text{if } \lambda \neq 0, x \geq 0 \\ \ln{(x+1)} & \text{if } \lambda = 0, x \geq 0 \\ -((-x+1)^{2-\lambda} - 1)/(2-\lambda) & \text{if } \lambda \neq 2, x < 0 \\ -\ln{(-x+1)} & \text{if } \lambda = 2, x < 0 \end{cases}')
    
    extender_gaussian_check = st.expander("Check the data distribution")
    with extender_gaussian_check:
        st.header("Gaussian Distribution Check")
        column = st.selectbox("Select a numeric column", df.select_dtypes(include=['number']).columns)

        if column:
            df_cleaned = df.dropna(subset=[column])
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

            axes[0].set_title("Distribution Plot")
            sns.histplot(df_cleaned[column], kde=True, ax=axes[0])
            axes[1].set_title("QQ Plot for Normality Check")
            sm.qqplot(df_cleaned[column], line='45', fit=True, ax=axes[1])
            plt.tight_layout()
            st.pyplot(fig)

            # Perform Shapiro-wilk test for normality
            st.write("Shapiro-Wilk Test for Normality")
            _, p_value = stats.shapiro(df_cleaned[column])
            st.write(f"P-value: {p_value}")
            if p_value < 0.05:
                st.warning("The data does not follow a normal distribution.")
            else:
                st.success("The data follows a normal distribution.")
        else:
            st.warning("Please select a numeric column for analysis.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    
    technique = st.selectbox("Select the technique", ["None", "Standardization", "MinMax Scaler", "MaxAbs Scaler", "Robust Scaler", "Quantile Transformer Scaler", "Box-Cox Transformation", "Yeo-Johnson Transformation"])
    col = st.multiselect("Select the column", df.select_dtypes(include=['number']).columns)
    
    if col:
        scaled_df = df.copy()

        if technique == "Standardization":
            scaler = StandardScaler()
        elif technique == "Normalization":
            scaler = MinMaxScaler()
        elif technique == "Robust Scaler":
            scaler = RobustScaler()
        elif technique == "MinMax Scaler":
            scaler = MinMaxScaler()
        elif technique == "MaxAbs Scaler":
            scaler = MaxAbsScaler()
        elif technique == "Quantile Transformer Scaler":
            scaler = QuantileTransformer(output_distribution='uniform')
        elif technique == "Box-Cox Transformation":
            scaler = PowerTransformer(method='box-cox')
        elif technique == "Yeo-Johnson Transformation":
            scaler = PowerTransformer(method='yeo-johnson')
            

        if technique != "None":
            for column in col:
                scaled_df[column] = scaler.fit_transform(np.array(scaled_df[column]).reshape(-1, 1))
        else:
            st.warning("No scaling technique selected.")
        
        expander_after_scaling = st.expander("Check the data distribution after scaling")
        with expander_after_scaling:
            st.write("Data after Feature Scaling:")
            for column in col:
                st.subheader(column+" Column:")
            # plot distribution and QQ plot
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

                # Plot distribution
                axes[0].set_title("Distribution Plot")
                sns.histplot(scaled_df[column], kde=True, ax=axes[0])

                # QQ plot for normality check
                axes[1].set_title("QQ Plot for Normality Check")
                sm.qqplot(scaled_df[column], line='45', fit=True, ax=axes[1])

                # Adjust layout
                plt.tight_layout()

                # Display the plots
                st.pyplot(fig)
                
                # Perform Kolmogorov-Smirnov test for normality
                st.write("Shapiro-Wilk Test for Normality")
                _, p_value = stats.shapiro(scaled_df[column])
                st.write(f"P-value: {p_value}")
                if p_value < 0.05:
                    st.warning("The data does not follow a normal distribution.")
                else:
                    st.success("The data follows a normal distribution.")
                    
            
            if st.checkbox("Confirm the Feature Scaling"):
                df = scaled_df
                st.session_state.df = df
        
    else:
        st.warning("Please select at least one numeric column for feature scaling.")
def feature_engineering(df):
    with tab2:
        expander_4 = st.expander("Feature Engineering")
        with expander_4:
            st.header("Feature Engineering")
            st.write("Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. This section allows you to create new features from existing features in the dataset.")
            st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        
    if expander_4.checkbox("Remove Missing Values"):
        remove_missing_values(df)
    if expander_4.checkbox("Missing Values Imputation"):
        missing_values_imputation(df)
    if expander_4.checkbox("Outliers Detection"):
        outliers_detection(df)
    if expander_4.checkbox("Feature Encoding"):
        feature_encoding(df)
    if expander_4.checkbox("Feature Scaling and Transformation"): 
        feature_scaling_transformation(df)


def pca(df):
    st.header("Principal Component Analysis (PCA)")
    st.write("Principal component analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset.")
    
    # Select numeric columns for PCA
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = st.multiselect("Select columns for PCA", numeric_columns)
    
    if selected_columns:
        # Standardize data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_columns])

        # Apply PCA
        pca_model = PCA()
        pca_result = pca_model.fit_transform(scaled_data)

        # Display explained variance ratio
        st.write("Explained Variance Ratio:")
        st.bar_chart(pca_model.explained_variance_ratio_)

        # Display scatter plot of principal components
        st.write("Scatter Plot of Principal Components:")
        st.scatter_chart(pd.DataFrame(pca_result, columns=['PC1', 'PC2']))
def tsne(df):

    st.header("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    st.write("t-distributed stochastic neighbor embedding (t-SNE) is a technique used to visualize high-dimensional data.")
    
    # Select numeric columns for t-SNE
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = st.multiselect("Select columns for t-SNE", numeric_columns)
    
    if selected_columns:
        # Apply t-SNE
        tsne_model = TSNE()
        tsne_result = tsne_model.fit_transform(df[selected_columns])

        # Display scatter plot of t-SNE results
        st.write("Scatter Plot of t-SNE:")
        st.scatter_chart(pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2']))
def umap(df):
        st.header("UMAP")
        st.write("Uniform manifold approximation and projection (UMAP) is a technique used to visualize high-dimensional data.")
def autoencoder(df):
    st.header("Autoencoder")
    st.write("Autoencoder is a type of neural network used to reduce the number of features in the dataset.")
    
    # Select numeric columns for autoencoder
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = st.multiselect("Select columns for Autoencoder", numeric_columns)
    
    if selected_columns:
        # Apply Autoencoder
        autoencoder_model = MLPRegressor(hidden_layer_sizes=[len(selected_columns)//2], max_iter=1000, random_state=42)
        autoencoder_model.fit(df[selected_columns], df[selected_columns])

        # Get reduced features from the hidden layer
        reduced_features = autoencoder_model.transform(df[selected_columns])

        # Display scatter plot of reduced features
        st.write("Scatter Plot of Reduced Features (from Autoencoder):")
        st.scatter_chart(pd.DataFrame(reduced_features, columns=['Feature1', 'Feature2']))
def feature_selection(df):
    with tab2:
        expander_5 = st.expander("Feature Selection & Dimensionality Reduction")
        with expander_5:
            st.header("Dimensionality Reduction")
            st.write("Dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. This section allows you to reduce the number of features in the dataset.")
    if expander_5.checkbox("PCA"):
        pca(df)
    if expander_5.checkbox("TSNE"):
        tsne(df)
    if expander_5.checkbox("UMAP"):
        umap(df)
    if expander_5.checkbox("Autoencoder"):
        autoencoder(df)


def Linear_Regression(X_train, X_test, y_train, y_test):
    st.write("Grid Search for Linear Regression")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    model = LinearRegression()
    scoring = st.selectbox("Select the scoring method", ["r2", "neg_mean_squared_error"])
    fit_intercept = st.multiselect("Select the fit_intercept method", [True, False])
    
    cv = st.slider("Select the number of folds", 2, 10, 5)
    param_grid = {
        'fit_intercept': fit_intercept
    }
    
    if st.checkbox("Tuning model"):
        
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
        grid_search.fit(X_train, y_train)
        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Mean Squared Error: {mean_squared_error(y_train, y_pred_train)}")
        st.write(f"R-squared: {r2_score(y_train, y_pred_train)}")

        st.write("Test Set:")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_test)}")
        st.write(f"R-squared: {r2_score(y_test, y_pred_test)}")
def Logistic_Regression(X_train, X_test, y_train, y_test):
    st.header("Grid Search for Logistic Regression")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    
    # User-selectable parameters
    penalty = st.multiselect("Select the regularization penalty", ['l1', 'l2', 'elasticnet', 'none'], ["l2"])
    C_range = st.slider("Select the range of regularization strength (C)", 0.01, 10.0, (4.0,5.0), step=0.01)
    fit_intercept = st.multiselect("Select the fit_intercept method", [True, False], [True])
    solver = st.multiselect("Select the optimization solver", ["liblinear", "lbfgs", "saga"], ["saga"])
    l1_ratio_range = st.slider("Select the range of L1 ratio", 0.0, 1.0, (0.4, 0.5), step=0.1)
    multi_class = st.selectbox("Select the multi-class strategy", ["auto", "ovr", "multinomial"])
    intercept_scaling_range = st.slider("Select the intercept scaling factor", 0.1, 10.0,(4.0, 5.0), step=0.01)
    
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)
    
    param_grid = {
        'penalty': penalty,
        'C': list(np.arange(C_range[0], C_range[1] + 0.1, 0.2)),
        'fit_intercept': fit_intercept,
        'solver': solver,
        'l1_ratio': list(np.arange(l1_ratio_range[0], l1_ratio_range[1]+0.1, 0.1)),
        'multi_class': [multi_class],
        'intercept_scaling': list(np.arange(intercept_scaling_range[0], intercept_scaling_range[1] + 0.1, 0.2))
    }
    
    if st.checkbox("Tuning model"):
        model = LogisticRegression()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)
        
        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    st.header("Grid Search for SVM")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    C_range = st.slider("Select the range of regularization strength (C)", 0.01, 10.0, (4.0, 5.0), step=0.01)
    kernel = st.multiselect("Select the kernel", ["linear", "poly", "rbf", "sigmoid"], ["rbf"])
    gamma_range = st.slider("Select the range of gamma", 0.01, 1.0, (0.1, 0.5), step=0.01)
    
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'kernel': kernel,
        'C': list(np.arange(C_range[0], C_range[1] + 0.1, 0.2)),
        'gamma': list(np.arange(gamma_range[0], gamma_range[1] + 0.01, 0.01))
    }

    if st.checkbox("Tuning model"):
        model = SVC()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def K_Nearest_Neighbour(X_train, X_test, y_train, y_test):
    st.header("Grid Search for KNN")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    n_neighbors = st.slider("Select the number of neighbors", 1, 20, 5)
    weights = st.multiselect("Select the weight function", ["uniform", "distance"], ["uniform"])
    algorithm = st.multiselect("Select the algorithm", ["auto", "ball_tree", "kd_tree", "brute"], ["auto"])
    p_value_range = st.slider("Select the power parameter for Minkowski distance", 1, 10,(4,5), 2)
    
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'n_neighbors': [n_neighbors],
        'weights': weights,
        'algorithm': algorithm,
        'p': list(np.arange(p_value_range[0], p_value_range[1] + 0.1, 0.2))
    }

    if st.checkbox("Tuning model"):
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def Decision_Tree(X_train, X_test, y_train, y_test):
    st.header("Grid Search for Decision Tree Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    criterion = st.multiselect("Select the criterion", ["gini", "entropy"], ["gini"])
    splitter = st.multiselect("Select the splitter", ["best", "random"], ["best"])
    max_depth_range = st.slider("Select the maximum depth", 1, 30, (5,15))
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)
    param_grid = {
        'criterion': criterion,
        'splitter': splitter,
        'max_depth': list(np.arange(max_depth_range[0], max_depth_range[1] + 1, 1))
    }

    if st.checkbox("Tuning model"):
        model = DecisionTreeClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def Random_Forest(X_train, X_test, y_train, y_test):
    st.header("Grid Search for Random Forest Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    n_estimators_range = st.slider("Select the number of trees", 1, 200, (100,150),step=10)
    criterion = st.multiselect("Select the criterion", ["gini", "entropy"], ["gini"])
    max_depth_range = st.slider("Select the maximum depth", 1, 20,(2,8))
    
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)
    param_grid = {
        'n_estimators': list(np.arange(n_estimators_range[0], n_estimators_range[1] + 1, 1)),
        'criterion': criterion,
        'max_depth': list(np.arange(max_depth_range[0], max_depth_range[1] + 1, 1))
    }

    if st.checkbox("Tuning model"):
        model = RandomForestClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def Ada_Boost(X_train, X_test, y_train, y_test):
    st.header("Grid Search for AdaBoost Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    n_estimators = st.slider("Select the number of estimators", 1, 200, 50)
    learning_rate = st.slider("Select the learning rate", 0.01, 1.0,(0.4, 0.5), step=0.1)
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'n_estimators': [n_estimators],
        'learning_rate': list(np.arange(learning_rate[0], learning_rate[1] + 1, 1))
    }

    if st.checkbox("Tuning model"):
        model = AdaBoostClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def Gradient_Boost(X_train, X_test, y_train, y_test):
    st.header("Grid Search for Gradient Boosting Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    n_estimators_range = st.slider("Select the number of estimators", 1, 200,(50,150))
    learning_rate_range = st.slider("Select the learning rate", 0.01, 1.0,(0.4,0.5), 0.1)
    max_depth_range = st.slider("Select the maximum depth", 1, 30,(5,15))
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'n_estimators': list(np.arange(n_estimators_range[0], n_estimators_range[1] + 1, 1)),
        'learning_rate': list(np.arange(learning_rate_range[0], learning_rate_range[1] + 1, 1)),
        'max_depth': list(np.arange(max_depth_range[0], max_depth_range[1] + 1, 1))
    }

    if st.checkbox("Tuning model"):
        model = GradientBoostingClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def XG_Boost(X_train, X_test, y_train, y_test):
    st.header("Grid Search for XGBoost Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    n_estimators_range = st.slider("Select the number of estimators", 1, 200,(50,150))
    learning_rate_range = st.slider("Select the learning rate", 0.01, 1.0,(0.4,0.5), 0.1)
    max_depth_range = st.slider("Select the maximum depth", 1, 30,(5,15))
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'n_estimators': list(np.arange(n_estimators_range[0], n_estimators_range[1] + 1, 1)),
        'learning_rate': list(np.arange(learning_rate_range[0], learning_rate_range[1] + 1, 1)),
        'max_depth': list(np.arange(max_depth_range[0], max_depth_range[1] + 1, 1))
    }

    if st.checkbox("Tuning model"):
        model = XGBClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def CatBoost(X_train, X_test, y_train, y_test):
    st.header("Grid Search for CatBoost Classifier")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    # User-selectable parameters
    iterations_range = st.slider("Select the number of iterations", 1, 100, (40,50))
    learning_rate_range = st.slider("Select the learning rate", 0.01, 1.0,(0.4,0.5), 0.1)
    max_depth_range = st.slider("Select the maximum depth", 1, 30,(5,15))
    cv = st.slider("Select the number of folds for cross-validation", 1, 11, 3)

    param_grid = {
        'iterations': list(np.arange(iterations_range[0], iterations_range[1] + 1, 1)),
        'learning_rate': list(np.arange(learning_rate_range[0], learning_rate_range[1] + 1, 1)),
        'max_depth': list(np.arange(max_depth_range[0], max_depth_range[1] + 1, 1))
    }
    
    if st.checkbox("Tuning model"):
        model = CatBoostClassifier()
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
        grid_search.fit(X_train, y_train)

        # best parameters
        best_params = grid_search.best_params_
        st.write("Best Parameters:", best_params)
        best_model = grid_search.best_estimator_
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        st.header("Model Evaluation:")
        st.write("Training Set:")
        st.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_train, y_pred_train))
        st.write("Classification Report:")
        st.text(classification_report(y_train, y_pred_train))

        st.write("Test Set:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_test))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred_test))
def model_train(X_train, X_test, y_train, y_test):
    st.header("Model Training")
    st.write("This section allows you to train a machine learning model to predict the target variable.")
    model = st.selectbox("Select the model", ["None", "Linear Regression", "Logistic Regression", "Support Vector Machine" , "K Nearest Neighbour", "Decision Tree", "Random Forest", "Ada Boost", "Gradient Boost", "XGBoost", "CatBoost"])
    
    if model == "None":
        pass
    
    elif model == "Linear Regression":
        Linear_Regression(X_train, X_test, y_train, y_test)
    elif model == "Logistic Regression":
        Logistic_Regression(X_train, X_test, y_train, y_test)
    elif model == "Support Vector Machine":
        Support_Vector_Machine(X_train, X_test, y_train, y_test)
    elif model == "K Nearest Neighbour":
        K_Nearest_Neighbour(X_train, X_test, y_train, y_test)
    elif model == "Decision Tree":
        Decision_Tree(X_train, X_test, y_train, y_test)
    elif model == "Random Forest":
        Random_Forest(X_train, X_test, y_train, y_test)
    elif model == "Ada Boost":
        Ada_Boost(X_train, X_test, y_train, y_test)
    elif model == "Gradient Boost":
        Gradient_Boost(X_train, X_test, y_train, y_test)
    elif model == "XGBoost":
        XG_Boost(X_train, X_test, y_train, y_test)
    elif model == "CatBoost":
        CatBoost(X_train, X_test, y_train, y_test)
        
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def model_building(df):
    with tab2:
        extender_6 = st.expander("Model Building")
        with extender_6:
            st.header("Model Building")
            st.write("This section allows you to build a machine learning model to predict the target variable.")
            st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
            st.write("Select the target variable:")
            target = st.selectbox("Select the target variable", df.columns)
            test_size = st.slider("Select the test size", 0.0, 1.0, 0.2, step=0.01)
            split = st.checkbox("Train_Test Split")
    if split:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=test_size, random_state=42)
        model_train(X_train, X_test, y_train, y_test)


def app():
    df = uplode_and_reset()
    
    with tab1:
        sidebar(df)

    # Section 1: Know your data
    know_data(df)
    
    # Section 2: Clean your data
    clean_data(df)
    
    # Section 3: Plot your data
    plot_data(df)
    
    # Section 4: Feature Engineering
    feature_engineering(df)
    
    # Section 5: Feature Selection
    feature_selection(df)
    
    # Section 6: Model Building
    model_building(df)
    
    # sidebar of the page


if __name__ == "__main__":
    st.set_page_config(layout="wide",initial_sidebar_state="auto")
    tab1,tab2 = st.sidebar.tabs(['Overview','Operations'])
    app()
    st.set_option('deprecation.showPyplotGlobalUse', False)
