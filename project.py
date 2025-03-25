# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Project1

#Koustav Ghosh
#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the path to your CSV file
file_path = r'C:\Users\Koustav\OneDrive\Desktop\Finlatics Internship\DsResearch\Media and Technology\Media and Technology\Global YouTube Statistics.csv'

# Try reading the CSV file with different encodings
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp1252')

# Display the DataFrame
print(df)



#Question1

# Sort the DataFrame by the 'Subscribers' column in descending order
df_sorted = df.sort_values(by='subscribers', ascending=False)

# Select the top 10 YouTube channels
top_10_channels = df_sorted.head(10)

# Display the top 10 YouTube channels
print(top_10_channels)



#Question2

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy = df[['category', 'subscribers']].copy()

# Ensure 'subscribers' column is string, fill missing values with '0'
df_copy['subscribers'] = df_copy['subscribers'].fillna('0').astype(str)

# Remove commas and convert to numeric
df_copy['subscribers'] = df_copy['subscribers'].str.replace(',', '')

# Convert to float first to handle cases with decimal points, then to int
df_copy['subscribers'] = df_copy['subscribers'].astype(float).astype(int)

# Group by 'category' and calculate the mean number of subscribers
category_avg_subscribers = df_copy.groupby('category')['subscribers'].mean()

# Find the category with the highest average number of subscribers
max_avg_subscribers_category = category_avg_subscribers.idxmax()
max_avg_subscribers_value = category_avg_subscribers.max()

print(f"The category with the highest average number of subscribers is '{max_avg_subscribers_category}' with an average of {max_avg_subscribers_value} subscribers.")




#Question3

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_uploads = df[['category', 'uploads']].copy()

# Ensure 'uploads' column is string, fill missing values with '0'
df_copy_uploads['uploads'] = df_copy_uploads['uploads'].fillna('0').astype(str)

# Remove commas and convert to numeric
df_copy_uploads['uploads'] = df_copy_uploads['uploads'].str.replace(',', '')

# Convert to float first to handle cases with decimal points, then to int
df_copy_uploads['uploads'] = df_copy_uploads['uploads'].astype(float).astype(int)

# Group by 'category' and calculate the mean number of uploads
category_avg_uploads = df_copy_uploads.groupby('category')['uploads'].mean()

print("Average number of videos uploaded by YouTube channels in each category:")
print(category_avg_uploads)




#Question4

# Create a copy of the relevant column to avoid changing the original DataFrame
df_copy_country = df[['Country']].copy()

# Ensure 'Country' column has no missing values
df_copy_country = df_copy_country.dropna()

# Count the number of YouTube channels per country
country_channel_counts = df_copy_country['Country'].value_counts()

# Get the top 5 countries with the highest number of YouTube channels
top_5_countries = country_channel_counts.head(5)

print("Top 5 countries with the highest number of YouTube channels:")
print(top_5_countries)




#Question5

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_distribution = df[['category', 'channel_type']].copy()

# Ensure no missing values in the columns
df_copy_distribution = df_copy_distribution.dropna()

# Create a pivot table to find the distribution of channel types across different categories
distribution = pd.crosstab(df_copy_distribution['category'], df_copy_distribution['channel_type'])

print("Distribution of channel types across different categories:")
print(distribution)





#Question6

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_corr = df[['subscribers', 'video views']].copy()

# Ensure there are no missing values
df_copy_corr = df_copy_corr.dropna()

# Convert the columns to strings, replace commas, and convert to numeric
df_copy_corr['subscribers'] = df_copy_corr['subscribers'].astype(str).str.replace(',', '').astype(float)
df_copy_corr['video views'] = df_copy_corr['video views'].astype(str).str.replace(',', '').astype(float)

# Calculate the correlation between the number of subscribers and total video views
correlation = df_copy_corr.corr().loc['subscribers', 'video views']

print("Correlation between the number of subscribers and total video views:", correlation)




#Question7

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_earnings = df[['category', 'lowest_monthly_earnings', 'highest_monthly_earnings']].copy()

# Ensure there are no missing values
df_copy_earnings = df_copy_earnings.dropna()

# Convert the columns to numeric, if necessary
df_copy_earnings['lowest_monthly_earnings'] = df_copy_earnings['lowest_monthly_earnings'].astype(str).str.replace(',', '').astype(float)
df_copy_earnings['highest_monthly_earnings'] = df_copy_earnings['highest_monthly_earnings'].astype(str).str.replace(',', '').astype(float)

# Calculate the average monthly earnings
df_copy_earnings['average_monthly_earnings'] = (df_copy_earnings['lowest_monthly_earnings'] + df_copy_earnings['highest_monthly_earnings']) / 2

# Group by category and calculate the mean of the average monthly earnings
category_earnings = df_copy_earnings.groupby('category')['average_monthly_earnings'].mean().sort_values(ascending=False)

print("Average Monthly Earnings by Category:")
print(category_earnings)



#Question8

# Create a copy of the relevant column to avoid changing the original DataFrame
df_copy_subscribers = df[['subscribers_for_last_30_days']].copy()

# Ensure there are no missing values
df_copy_subscribers = df_copy_subscribers.dropna()

# Convert the 'subscribers_for_last_30_days' column to numeric, if necessary
df_copy_subscribers['subscribers_for_last_30_days'] = df_copy_subscribers['subscribers_for_last_30_days'].astype(str).str.replace(',', '').astype(float)

# Plot the overall trend in subscribers gained in the last 30 days across all channels
plt.figure(figsize=(12, 6))
plt.hist(df_copy_subscribers['subscribers_for_last_30_days'], bins=30, color='skyblue', edgecolor='black')
plt.title('Overall Trend in Subscribers Gained in the Last 30 Days')
plt.xlabel('Subscribers Gained')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate and print summary statistics
mean_subscribers = df_copy_subscribers['subscribers_for_last_30_days'].mean()
median_subscribers = df_copy_subscribers['subscribers_for_last_30_days'].median()
max_subscribers = df_copy_subscribers['subscribers_for_last_30_days'].max()
min_subscribers = df_copy_subscribers['subscribers_for_last_30_days'].min()

print(f"Mean subscribers gained in the last 30 days: {mean_subscribers}")
print(f"Median subscribers gained in the last 30 days: {median_subscribers}")
print(f"Maximum subscribers gained in the last 30 days: {max_subscribers}")
print(f"Minimum subscribers gained in the last 30 days: {min_subscribers}")



#Question9


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_earnings = df[['lowest_yearly_earnings', 'highest_yearly_earnings']].copy()

# Ensure there are no missing values
df_copy_earnings = df_copy_earnings.dropna()

# Convert the earnings columns to numeric, if necessary
df_copy_earnings['lowest_yearly_earnings'] = df_copy_earnings['lowest_yearly_earnings'].astype(str).str.replace('[\$,]', '', regex=True).astype(float)
df_copy_earnings['highest_yearly_earnings'] = df_copy_earnings['highest_yearly_earnings'].astype(str).str.replace('[\$,]', '', regex=True).astype(float)

# Calculate the IQR for the highest yearly earnings
Q1 = df_copy_earnings['highest_yearly_earnings'].quantile(0.25)
Q3 = df_copy_earnings['highest_yearly_earnings'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = df_copy_earnings[(df_copy_earnings['highest_yearly_earnings'] < (Q1 - 1.5 * IQR)) | (df_copy_earnings['highest_yearly_earnings'] > (Q3 + 1.5 * IQR))]

# Plot the distribution and mark the outliers
plt.figure(figsize=(12, 6))
plt.boxplot(df_copy_earnings['highest_yearly_earnings'], vert=False)
plt.title('Boxplot of Highest Yearly Earnings')
plt.xlabel('Yearly Earnings')
plt.grid(True)
plt.show()

# Display the outliers
print("Outliers in terms of highest yearly earnings:")
print(outliers)



#Question10


# Create a copy of the relevant column to avoid changing the original DataFrame
df_copy_dates = df[['created_date']].copy()

# Convert the 'created_date' column to datetime
df_copy_dates['created_date'] = pd.to_datetime(df_copy_dates['created_date'], errors='coerce')

# Drop any rows with invalid dates
df_copy_dates = df_copy_dates.dropna(subset=['created_date'])

# Extract the year and month from the 'created_date'
df_copy_dates['year'] = df_copy_dates['created_date'].dt.year
df_copy_dates['month'] = df_copy_dates['created_date'].dt.month

# Plot the distribution of channel creation dates by year
plt.figure(figsize=(12, 6))
df_copy_dates['year'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of YouTube Channel Creation Dates by Year')
plt.xlabel('Year')
plt.ylabel('Number of Channels Created')
plt.grid(True)
plt.show()

# Plot the distribution of channel creation dates by month
plt.figure(figsize=(12, 6))
df_copy_dates['month'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of YouTube Channel Creation Dates by Month')
plt.xlabel('Month')
plt.ylabel('Number of Channels Created')
plt.grid(True)
plt.show()

# Plot the trend over time (yearly)
plt.figure(figsize=(12, 6))
df_copy_dates['year'].value_counts().sort_index().plot(kind='line')
plt.title('Trend of YouTube Channel Creation Over Time (Yearly)')
plt.xlabel('Year')
plt.ylabel('Number of Channels Created')
plt.grid(True)
plt.show()

# Plot the trend over time (monthly)
plt.figure(figsize=(12, 6))
df_copy_dates.set_index('created_date').resample('M').size().plot(kind='line')
plt.title('Trend of YouTube Channel Creation Over Time (Monthly)')
plt.xlabel('Date')
plt.ylabel('Number of Channels Created')
plt.grid(True)
plt.show()



#Question11


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_edu_channels = df[['Country', 'Gross tertiary education enrollment (%)']].copy()

# Ensure the 'Gross tertiary education enrollment (%)' is numeric
df_copy_edu_channels['Gross tertiary education enrollment (%)'] = pd.to_numeric(df_copy_edu_channels['Gross tertiary education enrollment (%)'], errors='coerce')

# Drop any rows with NaN values in the relevant columns
df_copy_edu_channels = df_copy_edu_channels.dropna(subset=['Country', 'Gross tertiary education enrollment (%)'])

# Count the number of YouTube channels per country
country_channel_counts = df['Country'].value_counts().reset_index()
country_channel_counts.columns = ['Country', 'Channel Count']

# Merge the two DataFrames on the 'Country' column
merged_df = pd.merge(df_copy_edu_channels, country_channel_counts, on='Country', how='inner')

# Plot the relationship between gross tertiary education enrollment and the number of YouTube channels
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='Gross tertiary education enrollment (%)', y='Channel Count')
plt.title('Relationship Between Gross Tertiary Education Enrollment and Number of YouTube Channels')
plt.xlabel('Gross Tertiary Education Enrollment (%)')
plt.ylabel('Number of YouTube Channels')
plt.grid(True)
plt.show()

# Calculate the correlation coefficient
correlation = merged_df['Gross tertiary education enrollment (%)'].corr(merged_df['Channel Count'])
print(f'Correlation coefficient: {correlation}')



#Question12


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_unemployment = df[['Country', 'Unemployment rate']].copy()

# Ensure the 'Unemployment rate' is numeric
df_copy_unemployment['Unemployment rate'] = pd.to_numeric(df_copy_unemployment['Unemployment rate'], errors='coerce')

# Drop any rows with NaN values in the relevant columns
df_copy_unemployment = df_copy_unemployment.dropna(subset=['Country', 'Unemployment rate'])

# Count the number of YouTube channels per country
country_channel_counts = df['Country'].value_counts().reset_index()
country_channel_counts.columns = ['Country', 'Channel Count']

# Get the top 10 countries with the highest number of YouTube channels
top_10_countries = country_channel_counts.head(10)

# Merge the top 10 countries with the unemployment data
merged_df = pd.merge(top_10_countries, df_copy_unemployment, on='Country', how='inner')

# Plot the unemployment rates for the top 10 countries
plt.figure(figsize=(12, 6))
plt.bar(merged_df['Country'], merged_df['Unemployment rate'], color='skyblue')
plt.title('Unemployment Rate Among Top 10 Countries with the Highest Number of YouTube Channels')
plt.xlabel('Country')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Print the merged DataFrame for verification
print(merged_df)



#Question13


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_urban = df[['Country', 'Urban_population']].copy()

# Ensure the 'Urban_population' is numeric
df_copy_urban['Urban_population'] = pd.to_numeric(df_copy_urban['Urban_population'], errors='coerce')

# Drop any rows with NaN values in the relevant columns
df_copy_urban = df_copy_urban.dropna(subset=['Urban_population'])

# Compute the average urban population percentage
average_urban_population = df_copy_urban['Urban_population'].mean()

print(f'The average urban population percentage in countries with YouTube channels is {average_urban_population:.2f}%')



#Question14


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_geo = df[['Latitude', 'Longitude']].copy()

# Ensure the 'Latitude' and 'Longitude' are numeric
df_copy_geo['Latitude'] = pd.to_numeric(df_copy_geo['Latitude'], errors='coerce')
df_copy_geo['Longitude'] = pd.to_numeric(df_copy_geo['Longitude'], errors='coerce')

# Drop any rows with NaN values in the relevant columns
df_copy_geo = df_copy_geo.dropna(subset=['Latitude', 'Longitude'])

# Generate a scatter plot to visualize the distribution
plt.figure(figsize=(10, 6))
plt.scatter(df_copy_geo['Longitude'], df_copy_geo['Latitude'], alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Geographical Distribution of YouTube Channels')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()



#Question15

# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_corr_pop = df[['Country', 'subscribers', 'Population']].copy()

# Ensure the 'subscribers' column is treated as string to replace commas, then convert to float
df_copy_corr_pop['subscribers'] = df_copy_corr_pop['subscribers'].astype(str).str.replace(',', '').astype(float)
df_copy_corr_pop['Population'] = pd.to_numeric(df_copy_corr_pop['Population'], errors='coerce')

# Drop any rows with NaN values in the relevant columns
df_copy_corr_pop = df_copy_corr_pop.dropna(subset=['subscribers', 'Population'])

# Group by country and aggregate the data
df_agg = df_copy_corr_pop.groupby('Country').agg({
    'subscribers': 'sum',
    'Population': 'mean'  # Assuming we want the average population for correlation
}).reset_index()

# Calculate the correlation between 'subscribers' and 'Population'
correlation = df_agg['subscribers'].corr(df_agg['Population'])

print(f"The correlation between the number of subscribers and the population of a country is: {correlation}")



#Question16


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_pop = df[['Country', 'Population']].copy()

# Drop any rows with NaN values in the relevant columns
df_copy_pop = df_copy_pop.dropna(subset=['Country', 'Population'])

# Ensure Population is numeric
df_copy_pop['Population'] = pd.to_numeric(df_copy_pop['Population'], errors='coerce')

# Count the number of YouTube channels per country
country_counts = df_copy_pop['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Channel_Count']

# Get the top 10 countries with the highest number of YouTube channels
top_10_countries = country_counts.head(10)

# Merge the top 10 countries with the original DataFrame to get their populations
top_10_countries_pop = top_10_countries.merge(df_copy_pop, on='Country', how='left')

# Aggregate the total population for these top 10 countries
top_10_countries_agg = top_10_countries_pop.groupby('Country').agg({
    'Channel_Count': 'first',  # Since each country appears once in top_10_countries
    'Population': 'mean'  # Assuming we want the average population for comparison
}).reset_index()

# Display the result
print(top_10_countries_agg)

# Plot the comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.bar(top_10_countries_agg['Country'], top_10_countries_agg['Population'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Total Population')
plt.title('Top 10 Countries with Highest Number of YouTube Channels and Their Total Population')
plt.xticks(rotation=45)
plt.show()



#Question17


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_corr_subs_unemploy = df[['Country', 'subscribers_for_last_30_days', 'Unemployment rate']].copy()

# Drop any rows with NaN values in the relevant columns
df_copy_corr_subs_unemploy = df_copy_corr_subs_unemploy.dropna(subset=['subscribers_for_last_30_days', 'Unemployment rate'])

# Ensure the relevant columns are numeric
if df_copy_corr_subs_unemploy['subscribers_for_last_30_days'].dtype == 'object':
    df_copy_corr_subs_unemploy['subscribers_for_last_30_days'] = df_copy_corr_subs_unemploy['subscribers_for_last_30_days'].str.replace(',', '').astype(float)

if df_copy_corr_subs_unemploy['Unemployment rate'].dtype == 'object':
    df_copy_corr_subs_unemploy['Unemployment rate'] = df_copy_corr_subs_unemploy['Unemployment rate'].str.replace(',', '').astype(float)

# Drop any rows with NaN values after conversion
df_copy_corr_subs_unemploy = df_copy_corr_subs_unemploy.dropna()

# Calculate the correlation
correlation = df_copy_corr_subs_unemploy['subscribers_for_last_30_days'].corr(df_copy_corr_subs_unemploy['Unemployment rate'])

# Display the correlation
print(f"The correlation between subscribers gained in the last 30 days and the unemployment rate is: {correlation}")

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_copy_corr_subs_unemploy['subscribers_for_last_30_days'], y=df_copy_corr_subs_unemploy['Unemployment rate'])
plt.xlabel('Subscribers Gained in the Last 30 Days')
plt.ylabel('Unemployment Rate')
plt.title('Correlation between Subscribers Gained in the Last 30 Days and Unemployment Rate')
plt.show()



#Question18


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_video_views = df[['channel_type', 'video_views_for_the_last_30_days']].copy()

# Drop any rows with NaN values in the relevant columns
df_copy_video_views = df_copy_video_views.dropna(subset=['channel_type', 'video_views_for_the_last_30_days'])

# Ensure the 'video_views_for_the_last_30_days' column is numeric
if df_copy_video_views['video_views_for_the_last_30_days'].dtype == 'object':
    df_copy_video_views['video_views_for_the_last_30_days'] = pd.to_numeric(df_copy_video_views['video_views_for_the_last_30_days'].str.replace(',', ''), errors='coerce')

# Drop any rows with NaN values after conversion
df_copy_video_views = df_copy_video_views.dropna()

# Plot the distribution
plt.figure(figsize=(14, 8))
sns.boxplot(x='channel_type', y='video_views_for_the_last_30_days', data=df_copy_video_views)
plt.xticks(rotation=90)
plt.xlabel('Channel Type')
plt.ylabel('Video Views for the Last 30 Days')
plt.title('Distribution of Video Views for the Last 30 Days across Different Channel Types')
plt.show()




#Question19



# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_videos = df[['created_month', 'uploads']].copy()

# Drop any rows with NaN values in the relevant columns
df_copy_videos = df_copy_videos.dropna(subset=['created_month', 'uploads'])

# Ensure 'created_month' column is numeric
df_copy_videos['created_month'] = pd.to_numeric(df_copy_videos['created_month'], errors='coerce')

# Drop any rows with NaN values after conversion
df_copy_videos = df_copy_videos.dropna()

# Group by month and calculate the average number of uploads
monthly_avg_uploads = df_copy_videos.groupby('created_month')['uploads'].mean().reset_index()


# Plotting the seasonal trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='created_month', y='uploads', data=monthly_avg_uploads, marker='o')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month')
plt.ylabel('Average Number of Uploads')
plt.title('Seasonal Trends in Number of Videos Uploaded by YouTube Channels')
plt.grid(True)
plt.show()




#Question20


# Create a copy of the relevant columns to avoid changing the original DataFrame
df_copy_subs = df[['created_date', 'subscribers_for_last_30_days']].copy()

# Drop any rows with NaN values in the relevant columns
df_copy_subs = df_copy_subs.dropna(subset=['created_date', 'subscribers_for_last_30_days'])

# Convert 'created_date' to datetime format
df_copy_subs['created_date'] = pd.to_datetime(df_copy_subs['created_date'], errors='coerce')

# Drop any rows with NaN values after conversion
df_copy_subs = df_copy_subs.dropna()

# Calculate the number of months since channel creation till now
df_copy_subs['months_since_creation'] = ((pd.Timestamp.now() - df_copy_subs['created_date']) / pd.Timedelta(days=30)).astype(int)

# Calculate average subscribers gained per month
average_subs_per_month = df_copy_subs['subscribers_for_last_30_days'].sum() / df_copy_subs['months_since_creation'].max()

print(f"Average number of subscribers gained per month since channel creation till now: {average_subs_per_month:.2f}")