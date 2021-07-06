# Using word2vec, t-SNE, and PCA to Predict Article Success
# Paper: Moniz, N and Torgo, L. (2018), "Multi-Source Social Feedback of
# Online News Feeds," ResearchGate.
# Data Source: https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms
# Contains titles and headlines from 93,239 news articles shared on Facebook, 
# LinkedIn, and GooglePlus, along with their sentiment scores

# Load data 

import pandas as pd
df = pd.read_csv('/Users/netisheth/Documents/TextAnalytics/News_SocialMedia.csv')
df.shape                                 # 93,239 x 11
df.columns

# Check correlation between SentimentTitle and SentimentHeadline
df[['Title', 'Headline']]
df.corr(method='pearson')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
polarity = []

for t in df.Title:
    sentiment = analyzer.polarity_scores(t)
    polarity.append(sentiment['compound'])

len(polarity)

import numpy as np
np.corrcoef(polarity, df.SentimentTitle)

df['Polarity'] = polarity
df.columns

# Convert PublishDate column to make it compatible with other datetime objects
# Earliest date in the dataset is 2002
df['temp'] = pd.datetime.now() - pd.to_datetime(df['PublishDate'])
df['DaysSincePub'] = df['temp'].astype('timedelta64[D]')
df = df.drop(['IDLink', 'Headline', 'PublishDate', 'SentimentHeadline', 'temp'], axis=1)

# Remove all rows with missing values (-1) for Facebook, Googleplus, and
# Facebook, and those that have missing values for Title or Source field
df = df[(df.Facebook!=-1) & (df.GooglePlus!=-1) & (df.LinkedIn!=-1)]
df.shape                                 # 81,637 x 9
df = df.drop(df[df['Title'].isna()].index, axis=0)
df = df.drop(df[df['Source'].isna()].index, axis=0)
df.shape                                 # 81,417 x 9

# Preprocess data 

from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

clean_titles = []
for t in df['Title']:
    words = regexp_tokenize(t.lower(), r'[A-Za-z]+')
    words = [w for w in words if len(w)>1 and w not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(w) for w in words]
    clean_titles.append(' '.join(words))

len(clean_titles)
df['CleanTitle'] = clean_titles
df.shape                                 # 81,417 x 10
df.columns

all_titles = ' '.join(clean_titles)
all_words = all_titles.split()
len(all_words)                           # 538,540 words (after stopword removal)
tokens = set(all_words)
len(tokens)                              # 24,523 tokens

# Create a Word2Vec model for our corpus

import gensim.downloader as api
fasttext = api.load('fasttext-wiki-news-subwords-300') # 963 MB word2vec model takes time to load
fasttext.vector_size
len(fasttext.vocab)                      # 999,999
sorted(fasttext.vocab)
obama_vec = fasttext['obama']
obama_vec[:20]                           # Check first 20 components of "obama"

tokens_filtered = [t for t in tokens if t in fasttext.vocab]
len(tokens_filtered)                     # 20,138
vector_list = [fasttext[t] for t in tokens_filtered]
word_vec_zip = zip(tokens_filtered, vector_list)   # Combine words with their vectors
word_vec_dict = dict(word_vec_zip)       # Cast to a dict to convert it to a df
df_dict = pd.DataFrame.from_dict(word_vec_dict, orient='index')
df_dict.shape
df_dict.head(3)

# Dimensionality reduction using t-SNE
# t-SNE has several hyperparameters: perplexity (0-100; 5-50 is a good range), 
# learning_rate (1-400; default=200), n_components (required)
# Let's map the first 400 vectors (every 10th word) in our dictionary into two 
# components to plot them on a 2D scatterplot (data may have > 2 components)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=10)
tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=25)
tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=50)
tsne_df_dict = tsne.fit_transform(df_dict[:400])  

# Plot a scatterplot of word vectors, labeling every 10th word, and using 
# adjustText library to adjust text position to avoid overlapping text

# pip install adjustText
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

sns.set()
fig, ax = plt.subplots(figsize=(12, 9))
sns.scatterplot(tsne_df_dict[:,0], tsne_df_dict[:,1], alpha=0.5)
words = []                        # Initialize list of words
words_to_plot = list(np.arange(0, 400, 10))
for w in words_to_plot:           # Append words to list
    words.append(plt.text(tsne_df_dict[w,0], tsne_df_dict[w,1], df_dict.index[w], fontsize=12))    
adjust_text(words, force_points=0.4, force_text=0.4, expand_points=(2,1), 
    expand_text=(1,2), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
plt.show()                        # Plot text using adjust_text

# To find titles that cluster together, we may use Doc2Vec for each title, 
# which require a lengthy training process, or we can use a convenient trick - 
# the average embeddings of all word vectors in each title

doc_vecs = []
for t in df.CleanTitle:
    words = t.split()
    words = [w for w in words if len(w)> 1 and w in tokens_filtered]
    if len(words) > 0:
        mean_vec = np.mean(fasttext[words], axis=0)
        doc_vecs.append(mean_vec)
    else:
        print(t)
        df = df[(df.CleanTitle != t)]

df.shape                          # 81,409 x 10
len(doc_vecs)                     # 81,409
doc_vecs[0:3]
df = df.reset_index(drop=True)

# t-SNE for document vectors (for first 400 titles, label every 50th title)

tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=25)
tsne_df = tsne.fit_transform(doc_vecs[:400])

fig, ax = plt.subplots(figsize=(14, 10))
sns.scatterplot(tsne_df[:,0], tsne_df[:,1], alpha=0.5)
titles = []
titles_to_plot = list(np.arange(0, 400, 50)) 
for t in titles_to_plot:
    titles.append(plt.text(tsne_df[t, 0], tsne_df[t, 1], clean_titles[t], fontsize=12))
adjust_text(titles, force_points=0.4, force_text=0.4, expand_points=(2,1), 
    expand_text=(1,2), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
plt.show()

# Popularity analysis: Scatterpof lot title sentiment by popularity (number of shares) 
# for all articles on Facebook, GooglePlus, or Linkedin

fig, ax = plt.subplots(1, 3, figsize=(14, 10))
subplots = [a for a in ax]
platforms = ['Facebook', 'GooglePlus', 'LinkedIn']
colors = list(sns.husl_palette(10, h=.5)[1:4]) 
for platform, subplot, color in zip(platforms, subplots, colors):
    sns.scatterplot(x = df[platform], y = df['SentimentTitle'], ax=subplot, color=color)
fig.suptitle('Plot of Popularity by Title Sentiment', fontsize=24)
plt.show()

# Since we don't see any pattern between sentiment and number of shares, let's 
# try a regression plot on a random subset (=5000) of this data, plotting 
# SentimentTitle  vs log(Popularity) o visualize linear relationships

subsample = df.sample(5000)
fig, ax = plt.subplots(1, 3, figsize=(15, 10))
subplots = [a for a in ax]
for platform, subplot, color in zip(platforms, subplots, colors):
    sns.regplot(x=np.log(subsample[platform]+1), y=subsample['SentimentTitle'], 
    ax=subplot, color=color, scatter_kws={'alpha':0.5})
    subplot.set_xlabel('')           # Replace standard x-label with a subplot title
    subplot.set_title(platform, fontsize=18)
fig.suptitle('Plot of log(Popularity) by Title Sentiment', fontsize=24)
plt.show()

# Still no relationship between sentiment and shares. How about we replace 
# SentimentTitle with Polarity by document source?

# Get top 12 sources by number of articles

source_names = list(df['Source'].value_counts()[:12].index)
source_colors = list(sns.husl_palette(12, h=.5))

fig, ax = plt.subplots(4, 3, figsize=(20,15), sharex=True, sharey=True)
ax = ax.flatten()
for ax, source, color in zip(ax, source_names, source_colors):
    sns.distplot(df.loc[df['Source'] == source]['Polarity'],
        ax=ax, color=color, kde_kws={'shade':True})
    ax.set_title(source, fontsize=14)
    ax.set_xlabel('')  
plt.xlim(-0.75, 0.75)
plt.show()

# Overlay density plots for 12 sources on the same plot for comparison

fig, ax = plt.subplots(figsize=(12, 8))
for source, color in zip(source_names, source_colors):
    sns.distplot(df.loc[df['Source'] == source]['Polarity'],
        ax=ax, hist=False, label=source, color=color)
    ax.set_xlabel('')  
plt.xlim(-0.75, 0.75)
plt.show()

# All 12 sources have distributions centered around 0 with similar tails
# Plot log(popularity) by platform

fig, ax = plt.subplots(3, 1, figsize=(15, 10))
subplots = [a for a in ax]
for platform, subplot, color in zip(platforms, subplots, colors):
    sns.distplot(np.log(df[platform] + 1), ax=subplot, color=color, kde_kws={'shade':True})
    subplot.set_title(platform, fontsize=18)
    subplot.set_xlabel('') 
fig.suptitle('Plot of log(Popularity) by Platform', fontsize=24)
plt.show()

# We see long-tailed distribution of articles on all three platforms

# Popularity prediction using PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=15, random_state=10)
pca_vecs = pca.fit_transform(doc_vecs)
len(pca_vecs)                              # 81,409
df_pca = pd.DataFrame(pca_vecs)
df_pca.shape                               # 81,409 x 15
df_pca.columns

# Combine the PCA matrix with article titles
df_pca = pd.concat((df_pca, df), axis=1)
df_pca.shape                              # 81,409 x 25
df_pca.columns                   

# Drop all non-numeric, non-dummy columns, for feeding into the models
df_pca = df_pca.drop(['Title', 'Source', 'Topic', 'SentimentTitle', 'CleanTitle'], axis=1)
df_pca.columns

# Subset Facebook data for analysis
facebook = df_pca.drop(columns=['GooglePlus', 'LinkedIn'])
Y = facebook['Facebook']
X = facebook.drop('Facebook', axis=1)

# Create training (80%) and test (20%) data sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Use the regression module in XGBoost to predict Facebook popularity
# XGBoost (Extreme Gradient Boost) is an ensemble ML algorithm which is fast, 
# parallizable, and outperforms other ensemble algorithms

!pip install xgboost
import xgboost as xgb
xgr = xgb.XGBRegressor(random_state=2)
xgr.fit(X_train, Y_train)
Y_pred = xgr.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, Y_pred)          # MSE=430,037

# MSE is not good. try hyperparameter tuning to improve results

from sklearn.model_selection import GridSearchCV
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], 'objective':['reg:linear'], 'max_depth': [5, 6, 7],
    'learning_rate': [.03, 0.05, .07], 'min_child_weight': [4], 'silent': [1],
    'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [250]}
xgb_grid = GridSearchCV(xgb1, parameters, cv=2, n_jobs=5, verbose=True)

xgb_grid.fit(X_train, Y_train)
print(xgb_grid.best_score_)  
print(xgb_grid.best_params_)

# GridSearchCV produces the following "best" model; rerun using this model
params = {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 
    'min_child_weight': 4, 'n_estimators': 250, 'nthread': 4, 
    'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}
xgr = xgb.XGBRegressor(random_state=2, **params)
xgr.fit(X_train, Y_train)
y_pred = xgr.predict(X_test)
mean_squared_error(Y_test, Y_pred)           # MSE=430,037

# No improvement in MSE. Data in its current format isn't working. Let's try some
# feature engineering by classifying aricles as duds (0 or 1 share) vs not dud

facebook['Facebook'].describe()

# Define a quick function that will return 1 (true) if the article has 0-1 shares
def dud_finder(popularity):
    if popularity <= 1:
        return 1
    else:
        return 0

# Create target column using the function
facebook['is_dud'] = facebook['Facebook'].apply(dud_finder)
facebook[['Facebook', 'is_dud']].head()
facebook['is_dud'].value_counts()
facebook['is_dud'].sum() / len(facebook)     # 28% of articles are duds

Y = facebook['is_dud']
X = facebook.drop('is_dud', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameter tuning
xgc = xgb.XGBClassifier()
parameters = {'nthread':[4], 'learning_rate': [.03, 0.05, .07],  'silent': [1],
    'max_depth': [5, 6, 7], 'min_child_weight': [4], 'subsample': [0.7],
    'colsample_bytree': [0.7], 'n_estimators': [100]}
xgb_grid = GridSearchCV(xgc, parameters, cv=2, n_jobs=5, verbose=True)

xgb_grid.fit(X_train, Y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

params = {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 
    'min_child_weight': 4, 'n_estimators': 200, 'nthread': 4, 'silent': 1, 
    'subsample': 0.7}
xgc = xgb.XGBClassifier(random_state=10, **params)

# Other ML classification algorithms

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=10)
knn = KNeighborsClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

preds = {}
for model_name, model in zip(['XGClassifier', 'RandomForestClassifier', 'KNearestNeighbors'], [xgc, rfc, knn]):
    model.fit(X_train, Y_train)
    preds[model_name] = model.predict(X_test)
    
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

for p in preds:
    print("{} performance:".format(p))
    print()
    print(classification_report(Y_test, preds[p]), sep='\n')

# Plot ROC curves
for model in [xgc, rfc, knn]:
    fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.show()

# Facebook popularity: Second try
# Use averaged probability that each article is a dud as a regressor

averaged_probs = (xgc.predict_proba(X)[:, 1] + knn.predict_proba(X)[:, 1] + 
     rfc.predict_proba(X)[:, 1]) / 3

Y = facebook['Facebook']
X['prob_dud'] = averaged_probs

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameter tuning
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], 'objective':['reg:linear'], 'max_depth': [5, 6, 7],
    'learning_rate': [.03, .05, .07], 'min_child_weight': [4], 'silent': [1],
    'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [250]}
xgb_grid = GridSearchCV(xgb1, parameters, cv = 2, n_jobs = 5, verbose=True)

xgb_grid.fit(X_train, Y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

params = {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 
          'min_child_weight': 4, 'n_estimators': 250, 'nthread': 4, 
          'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}
xgr = xgb.XGBRegressor(random_state=2, **params)

xgr.fit(X_train, Y_train)
Y_pred = xgr.predict(X_test)
mean_squared_error(Y_test, Y_pred)               # MSE=776

for feature, importance in zip(list(X.columns), xgr.feature_importances_):
    print('Model weight for feature {}: {}'.format(feature, importance))

# The model thought that prob_dud was the most important feature!                
