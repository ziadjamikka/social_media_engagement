# Ziad Elsayed Ebrahim
# Tasneem Samir Mohamed
# Marwan Wael El-Wageeh
# Mohamed Yasser Mohamed Hamdy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

plt.style.use('seaborn-v0_8')
colors_palette = plt.get_cmap('Set3').colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors_palette)

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['post_time'] = pd.to_datetime(df['post_time'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
    df['post_hour'] = df['post_time'].dt.hour
    df['sentiment_score'] = df['sentiment_score'].astype(str).str.lower()
    df['engagment'] = df['likes'] + df['comments'] + df['shares']
    df['total_engagement'] = df['engagment']
    return df

def basic_summary(df):
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    print(df.describe(include='all'))
    print(df.head())

def plot_distributions(df):
    for col in df.columns[1:]:
        plt.figure(figsize=(8, 4))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            df[col].value_counts().plot(kind='bar')
            plt.title(f'Value Counts of {col}')
        else:
            plt.text(0.5, 0.5, f'Cannot plot column: {col}', ha='center')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def engagement_statistics(df):
    result = df.groupby(['platform','post_type'])['engagment'].agg([np.mean, min, max]).sort_values(by="mean", ascending=False)
    print(result.head(10))

def boxplots_and_barplots(df):
    sns.boxplot(x=df['engagment'])
    plt.tight_layout()
    sns.boxplot(x='post_day', y='engagment', data=df)
    plt.title("Engagement by Day of Week")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    sns.barplot(data=df, x='post_type', y='engagment')
    plt.title("Engagement by Post Type")
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='platform', y='engagment', data=df, estimator=sum)
    plt.title("Total Engagement by Platform")
    plt.tight_layout()
    plt.show()

def heatmap_engagement(df):
    pivot = pd.pivot_table(df, values='engagment', index='platform', columns='post_type', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title("Average Engagement by Platform and Post Type")
    plt.tight_layout()
    plt.show()

def sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='platform', hue='sentiment_score', data=df)
    plt.title("Sentiment Distribution by Platform")
    plt.tight_layout()
    plt.show()

    sns.countplot(data=df, x='sentiment_score', order=df['sentiment_score'].value_counts().index)
    plt.title("Sentiment Distribution")
    plt.show()

def plotly_visualizations(df):
    fig1 = px.bar(
        df.groupby('platform')['total_engagement'].sum().reset_index().sort_values(by='total_engagement', ascending=False),
        x='platform',
        y='total_engagement',
        title='Total Engagement by Platform',
        text_auto=True,
        color='platform'
    )
    fig1.show()

    for col in ['likes', 'shares', 'comments']:
        fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col.capitalize()}')
        fig.show()

    avg_comments = df.groupby("platform")["comments"].mean().reset_index()
    fig = px.pie(
        avg_comments,
        values='comments',
        names='platform',
        title='Average Comments per Platform',
        width=400,
        height=400
    )
    fig.show()

    sentiment = df.groupby(["platform", "sentiment_score"]).size().reset_index(name='count')
    fig = px.bar(sentiment, x="platform", y="count", color="sentiment_score", title="Sentiment Distribution by Platform", barmode="stack")
    fig.show()

def time_series_and_grouped(df):
    df_sorted = df.sort_values(by='post_time')
    plt.plot(df_sorted['post_time'], df_sorted['likes'])
    plt.title("Likes Over Time")
    plt.xlabel("Date")
    plt.ylabel("Likes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    engagement = df.groupby("post_type")[["likes", "comments", "shares"]].mean().reset_index()
    engagement = pd.melt(engagement, id_vars="post_type", var_name="metric", value_name="avg_value")

    plt.figure(figsize=(10,6))
    sns.barplot(data=engagement, x="post_type", y="avg_value", hue="metric")
    plt.title("Average Engagement by Post Type")
    plt.xticks(rotation=45)
    plt.show()

    df.groupby('post_day')[['likes', 'comments', 'shares']].mean().plot(kind='bar')
    plt.title('Average Engagement per Day')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45)
    plt.show()

def main():
    df = load_and_prepare_data('social_media_engagement1.csv')
    basic_summary(df)
    plot_distributions(df)
    engagement_statistics(df)
    boxplots_and_barplots(df)
    heatmap_engagement(df)
    sentiment_distribution(df)
    plotly_visualizations(df)
    time_series_and_grouped(df)

if __name__ == "__main__":
    main()
