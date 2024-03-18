import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff



chartdata=pd.DataFrame(np.random.randn(40,3),columns=['l1','l2','l3'])

st.title("line")
st.line_chart(chartdata)

st.bar_chart(chartdata)

st.area_chart(chartdata)

st.title('Visualizations with Plotly')

# Load the dataset
df = pd.read_csv('iris.csv')

# Display the first few rows of the dataset
st.dataframe(df.head())

st.text('1. Pie Chart')
fig = px.pie(df, values='petal_length', names='species')
st.plotly_chart(fig)

st.text('2. Pie Chart with Hole')
fig = px.pie(df, values='petal_length', names='species', opacity=0.8, hole=0.5,
             color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig)


st.text('3. Multiple Dist_plots')
# Generate random data for the distribution plots
x1 = np.random.randn(200) + 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) - 2
hist_data = [x1, x2, x3]

group_labels = ['G1', 'G2', 'G3']

# Create distribution plots
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.1, 0.1])

# Display the distribution plots
st.plotly_chart(fig, use_container_width=True)
