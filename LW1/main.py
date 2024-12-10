import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Налаштування стилю візуалізації
sns.set(rc={'figure.figsize': (10, 8)})


def detect_outliers(data):
    """
    Визначає викиди за допомогою методу міжквартильного розмаху (IQR).
    :param data: Числові дані у вигляді масиву
    :return: Список індексів значень, які є викидами
    """
    data = np.array(data)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]


# Завантаження набору даних
df = pd.read_csv('../data/OnlineNewsPopularityReduced.csv', delimiter=',')

# Відображення інформації про набір даних
print(df.info())
print("\nОписова статистика по числовим змінним:")
print(df.describe())

# Вибір лише категоріальних змінних
categorical_columns = [col for col in df.columns if 'is_' in col or df[col].nunique() <= 10]

# Виведення описової статистики для кожної категоріальної змінної
for column in categorical_columns:
    print(f"Описова статистика по категоріальній змінній: {column}")
    print(df[column].describe())
    print("\n")


# Функція для візуалізації розподілів та виявлення викидів
def visualize_and_detect_outliers(column_name, bins=50, xlabel='', ylabel='Count'):
    df[column_name].hist(bins=bins)
    plt.title(f'Гістограма {column_name}')
    plt.xlabel(xlabel or column_name)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.show()

    outliers = detect_outliers(df[column_name])
    print(f"{column_name}: знайдено {len(outliers)} викидів.\n")
    return outliers


# Аналіз цільової змінної: 'shares'
shares_outliers = visualize_and_detect_outliers('shares', bins=100)

# Аналіз 'n_tokens_title'
n_tokens_title_outliers = visualize_and_detect_outliers('n_tokens_title', bins=50)

# Аналіз 'n_tokens_content'
n_tokens_content_outliers = visualize_and_detect_outliers('n_tokens_content', bins=50)

# Аналіз розподілу новин за днями тижня
news_count_by_weekdays = {
    day: df.loc[df[f'weekday_is_{day}'] == 1].shape[0]
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
}

most_news_day = max(news_count_by_weekdays, key=news_count_by_weekdays.get)
least_news_day = min(news_count_by_weekdays, key=news_count_by_weekdays.get)

print(f"Найбільше новин опубліковано у: {most_news_day} ({news_count_by_weekdays[most_news_day]})")
print(f"Найменше новин опубліковано у: {least_news_day} ({news_count_by_weekdays[least_news_day]})")

# Візуалізація розподілу новин за днями тижня
plt.bar(news_count_by_weekdays.keys(), news_count_by_weekdays.values(), color='skyblue')
plt.title('Розподіл новин за днями тижня')
plt.xlabel('Дні')
plt.ylabel('Кількість новин')
plt.show()


# Аналіз кореляцій
def correlation_analysis(column1, column2):
    correlation = df[column1].corr(df[column2])
    print(f"Кореляція між {column1} та {column2}: {correlation}\n")
    return correlation


# Аналіз 'n_tokens_title' vs 'shares'
correlation_analysis('n_tokens_title', 'shares')

# Аналіз впливу зображень та відео на популярність
imgs_correlation = correlation_analysis('num_imgs', 'shares')
videos_correlation = correlation_analysis('num_videos', 'shares')

# Порівняння популярності у вихідні та будні дні
weekend_shares_avg = df.loc[df['is_weekend'] == 1, 'shares'].mean()
weekday_shares_avg = df.loc[df['is_weekend'] == 0, 'shares'].mean()

plt.bar(['Вихідні', 'Будні'], [weekend_shares_avg, weekday_shares_avg], color=['skyblue', 'salmon'])
plt.title('Середня кількість поширень: вихідні проти буднів')
plt.xlabel('Категорія')
plt.ylabel('Середня кількість поширень')
for i, value in enumerate([weekend_shares_avg, weekday_shares_avg]):
    plt.text(i, value, f"{int(value)}", ha='center', va='bottom')
plt.show()

# Залежність між 'n_tokens_content' та 'shares'
plt.scatter(df['n_tokens_content'], df['shares'], alpha=0.5)
plt.title('n_tokens_content vs shares')
plt.xlabel('n_tokens_content')
plt.ylabel('Shares')
plt.grid(True)
plt.show()
correlation_analysis('n_tokens_content', 'shares')

# Аналіз популярності за категоріями каналів
data_channels = [
    'lifestyle', 'entertainment', 'bus',
    'socmed', 'tech', 'world'
]
shares_by_channels = {
    channel: df.loc[df[f'data_channel_is_{channel}'] == 1, 'shares'].mean()
    for channel in data_channels
}

plt.bar(shares_by_channels.keys(), shares_by_channels.values(), color='skyblue')
plt.title('Середня кількість поширень за категоріями каналів')
plt.xlabel('Категорія')
plt.ylabel('Кількість поширень')
plt.show()

# Аналіз впливу позитивних і негативних слів на популярність
correlation_analysis('rate_positive_words', 'shares')
correlation_analysis('rate_negative_words', 'shares')

print("На основі кореляцій вплив позитивних і негативних слів на популярність є незначним.\n")