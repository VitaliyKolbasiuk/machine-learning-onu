import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setting visualization style
sns.set(rc={'figure.figsize': (10, 8)})


def detect_outliers(data):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    :param data: Array-like numerical data
    :return: List of indices of outlier values
    """
    data = np.array(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]


# Load dataset
df = pd.read_csv('./data/OnlineNewsPopularityReduced.csv', delimiter=',')

# Display dataset information
print(df.info())
print("\nDescriptive statistics for numerical columns:")
print(df.describe())


# Define function for visualizing distributions and identifying outliers
def visualize_and_detect_outliers(column_name, bins=50, xlabel='', ylabel='Count'):
    df[column_name].hist(bins=bins)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(xlabel or column_name)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.show()

    outliers = detect_outliers(df[column_name])
    print(f"{column_name}: {len(outliers)} outliers detected.\n")
    return outliers


# Analyze target variable: 'shares'
shares_outliers = visualize_and_detect_outliers('shares', bins=100)

# Analyze 'n_tokens_title'
n_tokens_title_outliers = visualize_and_detect_outliers('n_tokens_title', bins=50)

# Analyze 'n_tokens_content'
n_tokens_content_outliers = visualize_and_detect_outliers('n_tokens_content', bins=50)

# Analyze news distribution by weekdays
news_count_by_weekdays = {
    day: df.loc[df[f'weekday_is_{day}'] == 1].shape[0]
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
}

most_news_day = max(news_count_by_weekdays, key=news_count_by_weekdays.get)
least_news_day = min(news_count_by_weekdays, key=news_count_by_weekdays.get)

print(f"Most news published on: {most_news_day} ({news_count_by_weekdays[most_news_day]})")
print(f"Least news published on: {least_news_day} ({news_count_by_weekdays[least_news_day]})")

# Visualize weekday distribution
plt.bar(news_count_by_weekdays.keys(), news_count_by_weekdays.values(), color='skyblue')
plt.title('News distribution by weekdays')
plt.xlabel('Days')
plt.ylabel('Count of news')
plt.show()


# Correlation analysis
def correlation_analysis(column1, column2):
    correlation = df[column1].corr(df[column2])
    print(f"Correlation between {column1} and {column2}: {correlation}\n")
    return correlation


# Analyze 'n_tokens_title' vs 'shares'
correlation_analysis('n_tokens_title', 'shares')

# Analyze impact of images and videos on popularity
imgs_correlation = correlation_analysis('num_imgs', 'shares')
videos_correlation = correlation_analysis('num_videos', 'shares')

# Compare weekend vs weekday shares
weekend_shares_avg = df.loc[df['is_weekend'] == 1, 'shares'].mean()
weekday_shares_avg = df.loc[df['is_weekend'] == 0, 'shares'].mean()

plt.bar(['Weekend', 'Not Weekend'], [weekend_shares_avg, weekday_shares_avg], color=['skyblue', 'salmon'])
plt.title('Average Shares: Weekend vs Not Weekend')
plt.xlabel('Category')
plt.ylabel('Average Shares')
for i, value in enumerate([weekend_shares_avg, weekday_shares_avg]):
    plt.text(i, value, f"{int(value)}", ha='center', va='bottom')
plt.show()

# Dependency between 'n_tokens_content' and 'shares'
plt.scatter(df['n_tokens_content'], df['shares'], alpha=0.5)
plt.title('n_tokens_content vs shares')
plt.xlabel('n_tokens_content')
plt.ylabel('Shares')
plt.grid(True)
plt.show()
correlation_analysis('n_tokens_content', 'shares')

# Analyze popularity by data channel
data_channels = [
    'lifestyle', 'entertainment', 'bus',
    'socmed', 'tech', 'world'
]
shares_by_channels = {
    channel: df.loc[df[f'data_channel_is_{channel}'] == 1, 'shares'].mean()
    for channel in data_channels
}

plt.bar(shares_by_channels.keys(), shares_by_channels.values(), color='skyblue')
plt.title('Average shares by data channel')
plt.xlabel('Channel')
plt.ylabel('Shares')
plt.show()

# Analyze impact of sentiment rates
correlation_analysis('rate_positive_words', 'shares')
correlation_analysis('rate_negative_words', 'shares')

print("Based on correlations, the impact of positive and negative words on shares is negligible.\n")
