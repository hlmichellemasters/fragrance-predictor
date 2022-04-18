from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import datasets, linear_model
from .models import Preference, User, Perfume
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


def build_model_for_user(user, reviews_df):
    perfumes_df = pd.DataFrame.from_records(Perfume.objects.all().values('id', 'name', 'house', 'description'))
    perfumes_df = add_features_to_perfume_dataframe(perfumes_df)

    # inner join the perfume and review data to include all important columns for reviewed perfumes
    perfume_reviews_df = pd.merge(reviews_df, perfumes_df, how='inner', left_on='perfume_id', right_on='id')

    # split the data for training and testing
    train_data, test_data, train_labels, test_labels = \
        train_test_split(perfume_reviews_df['features'].values.astype('U'), perfume_reviews_df['love'],
                         test_size=0.2, random_state=1)

    counter = CountVectorizer(stop_words='english')
    counter.fit(train_data)
    train_counts = counter.transform(train_data)
    test_counts = counter.transform(test_data)

    classifier = MultinomialNB()
    classifier.fit(train_counts, train_labels)

    predictions = classifier.predict(test_counts)

    accuracy = accuracy_score(test_labels, predictions)

    return classifier, accuracy, perfumes_df, perfume_reviews_df, counter


# create similarity score tuples [(perfume_id, similarity_score), (.,.), ...]
def create_similarity_matrix(perfume_id, perfume_data):
    # vectorize the features
    vectorizer = CountVectorizer().fit_transform(perfume_data['features'])

    # create the cosine similarity matrix from the features
    cs = cosine_similarity(vectorizer)

    scores = list(enumerate(cs[perfume_id]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # first one is itself (so 0 is excluded)
    return sorted_scores[1:11]


# function to find recommendations from the cold-start recommendation form
def find_perfumes_from_features(loves_notes, not_loves_notes, loves_df, not_loves_df):
    # create a loves dataframe to use to create the similarity matrix with rest of perfumes
    love_features = ""
    for i in range(0, loves_df.shape[0]):
        new_feature = ' ' + loves_df['name'][i] + ' ' + loves_df['house'][i] + ' ' + loves_df['description'][i]
        love_features += new_feature

    for word in loves_notes:
        love_features += word

    custom_loves_data = {'id': [0],
                         'name': [' '],
                         'house': [' '],
                         'description': [' '],
                         'features': love_features
                         }
    custom_loves_df = pd.DataFrame(custom_loves_data)

    # pull all the perfumes from database and add the features column
    all_perfumes_df = pd.DataFrame.from_records(Perfume.objects.all().values('id', 'name', 'house', 'description'))
    all_perfumes_df = add_features_to_perfume_dataframe(all_perfumes_df)

    # subtract all the perfumes that user has already mentioned (loved and not loved)
    unreviewed_df = all_perfumes_df[~all_perfumes_df.id.isin([custom_loves_df.id, not_loves_df.id])]
    # unreviewed_df = unreviewed_df[~unreviewed_df.id.isin([not_loves_df.id, custom_loves_df.id])]

    # subtract all the features that the user input they do not love
    not_loves_list = re.findall(r'\s|,|[^,\s]+', not_loves_notes)
    for feature in not_loves_list:
        unreviewed_df = unreviewed_df[~unreviewed_df['features'].str.contains(feature)]

    unreviewed_df = custom_loves_df.append(unreviewed_df)

    sorted_scores = create_similarity_matrix(0, unreviewed_df)

    print('Recommended perfumes based off of your inputs of loves: \'', loves_df, '\' are:\n')
    print('and your inputs of not loves: \'', not_loves_df, '\' and \'', not_loves_notes, '\' are:\n')

    # loop through the returned sorted_scores matrix (up to 20?) and then add the formatted score
    scores = []
    recommended_perfumes = []

    for row in sorted_scores:
        # recommended_perfume = unreviewed_df[unreviewed_df.id == row[0]]['name'].values[0]
        recommended_perfume = Perfume.objects.get(id=row[0])
        similarity_percent = row[1] * 100
        scores.append("{:.2f}".format(similarity_percent))
        recommended_perfumes.append(recommended_perfume)

    perfumes = zip(recommended_perfumes, scores)

    return perfumes


def add_features_to_perfume_dataframe(dataframe):
    features = []
    for i in range(0, dataframe.shape[0]):
        new_feature = ' ' + dataframe['name'][i] + ' ' + dataframe['house'][i] + ' ' + dataframe['description'][i]
        features.append(new_feature)

    dataframe['features'] = features

    return dataframe


def split_and_vectorize_data(data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data.values.astype('U'), labels, test_size=0.2,
                                                                        random_state=1)

    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer.fit(data.values.astype('U'))
    train_counts = count_vectorizer.transform(train_data)
    test_counts = count_vectorizer.transform(test_data)

    print("Num Train Data: " + str(len(train_data)))
    print("Num Test Data: " + str(len(test_data)))

    return count_vectorizer, train_counts, test_counts, train_labels, test_labels


def frequency_transform(counts):
    tfidf_transformer = TfidfTransformer()
    transformed_counts = tfidf_transformer.fit_transform(counts)
    return transformed_counts


def fit_multi_NB(train_counts, test_counts, train_labels, test_labels):
    classifier = MultinomialNB()
    classifier.fit(train_counts, train_labels)

    predictions = classifier.predict(test_counts)

    print("Accuracy score for multi NB: {0:.2f}".format(accuracy_score(test_labels, predictions)))

    return accuracy_score(test_labels, predictions)


def random_forest(train_counts, test_counts, train_labels, test_labels):
    forest = RandomForestClassifier()
    forest.fit(train_counts, train_labels)

    predictions = forest.predict(test_counts)

    print("Accuracy score for random forest: {0:.2f}".format(accuracy_score(test_labels, predictions)))


def scale(train_counts, test_counts):
    # Standardize the features
    train_counts = StandardScaler(with_mean=False).fit_transform(train_counts)
    test_counts = StandardScaler(with_mean=False).fit_transform(test_counts)

    return train_counts, test_counts


def cluster(counts, train_counts):
    NUMBER_OF_CLUSTERS = 2

    km = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        init='k-means++',
        max_iter=500)
    km.fit(train_counts)

    # First: for every document we get its corresponding cluster
    clusters = km.predict(counts)

    return clusters, km


def reduce(counts):
    pca = IncrementalPCA(n_components=2)
    two_dim = pca.fit_transform(counts.todense())
    print("Explained variance with PCA: " + str(pca.explained_variance_))

    scatter_x = two_dim[:, 0]  # first principle component
    scatter_y = two_dim[:, 1]  # second principle component

    return scatter_x, scatter_y


def truncate(counts):
    svd = TruncatedSVD(n_components=2, random_state=13)

    data = svd.fit_transform(counts)

    pca = IncrementalPCA(n_components=2)
    two_dim = pca.fit_transform(counts.todense())
    print("Explained variance with Truncated SVD: " + str(svd.explained_variance_))

    scatter_x = two_dim[:, 0]  # first principle component
    scatter_y = two_dim[:, 1]  # second principle component

    return scatter_x, scatter_y


# Plot it
def plot_PCA(clusters, scatter_x, scatter_y):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Principal Component ', fontsize=15)
    ax.set_ylabel('Second Principal Component ', fontsize=15)
    ax.set_title('Principal Component Analysis (2PCs)', fontsize=20)

    cmap = {0: 'red', 1: 'blue', 2: 'black', 3: 'green'}

    for group in np.unique(clusters):
        i = np.where(clusters == group)
        ax.scatter(scatter_x[i],
                   scatter_y[i],
                   c=cmap[group],
                   label=group)
    ax.legend()
    ax.grid()
    plt.show()


def plot_num_love(user_data):
    plt.figure(figsize=(5, 5))
    ax = sb.countplot(x=user_data['Love'], data=user_data, order=user_data['Love'].value_counts().index)
    for p, label in zip(ax.patches, user_data['Love'].value_counts()):
        ax.annotate(label, (p.get_x() + 0.25, p.get_height() + 0.5))
    plt.show()


def peek_at_clusters(count_vectorizer, km):
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = count_vectorizer.get_feature_names_out()
    stop_words = count_vectorizer.stop_words_
    #    for word in stop_words:
    #       print(stop_words)
    print(stop_words)
    for i in range(2):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


def linear_regression(x, y):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    # Train the model using the training sets
    regr.fit(x, y)

    # Make predictions using the testing set
    y_pred = regr.predict(x)

    return y_pred
