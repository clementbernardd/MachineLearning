# Importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")


# Merge of the two datasets


tmdb = pd.read_csv("../Data/TMDB/tmdb_5000_movies.csv")
tmdb2 = pd.read_csv("../Data/TMDB/tmdb_5000_credits.csv")
tmdb2.rename(columns={'movie_id':'id'}, inplace=True)
data = pd.merge(tmdb,tmdb2, on='id')


def drop_useless_data(ds) :
    # It drops instances where it misses some value
    # It drops also data where the time is lower than 1985

    # Drops vote count
    ds = ds.drop(ds.vote_count[ ds.vote_count < 10].index)
    # Drops overview
    index_overview_nan = ds.overview[ds.overview.isnull()].index
    ds = ds.drop(index_overview_nan)
    # Delete instances where the runtime is null
    ds = ds.drop(  ds.runtime[ds.runtime == 0].index)
    # Delete instances where the budget is null
    ds = ds.drop(  ds.budget[ds.budget == 0].index)
    # Delete instances where the revenue is null
    ds = ds.drop(  ds.revenue[ds.revenue == 0].index)
    # Reset the indexes
    ds = ds.reset_index()
    return ds


data = drop_useless_data(data)


# It seems that homepage and tagline have missing values
# Nevertheless, homepage is a string of the summary of the movie. We will take care of it only if we have time.
# Tagline gives kind of slogan of the movie. We don't take care of it now
# Then we convert the time

# Conversion of the time
def convert_date_time(ds,name) :
    # Returns new database where the string date time is converted to date type
    new_ds = ds.copy()
    new_ds[name] = pd.to_datetime(ds[name])
    return new_ds

data = convert_date_time(data,'release_date')

### Then we drop all the instances where the movies are inferior than 1985
### It seems obsolete for our criteria

def drop_time(ds) :
    new_ds = ds.copy()
    time = pd.to_datetime('1985-01-01')
    indexes = new_ds.release_date[new_ds.release_date < time].index
    new_ds = new_ds.drop(indexes)
    new_ds = new_ds.reset_index()
    return new_ds



data = drop_time(data)

### Create a feature for the year
### 1-of-K encoding for the months

def encode_year(ds) :
    # Creates a new column with the year
    X = ds.copy()
    X['year'] = ds.release_date.apply(lambda x : x.year)
    return X
def encode_month(ds) :
    # Creates new columns for months
    X = ds.copy()
    X['month'] = X.release_date.apply(lambda x : x.month)
    new_col = pd.get_dummies(X['month'])
    columns_name = {float(i) : 'month'+'_'+str(i) for i in range(1,13)}
    new_col.rename(columns = columns_name, inplace=True)
    ds_conc = pd.concat([X, new_col], axis=1)
    return ds_conc




data = encode_year(data)
data =encode_month(data)



### Vizualisation of the data in order to convert it properly

import json
def get_json(ds,name) :
    # Return a list with the different json file in the column name and the count of it
    genres = []
    count = []
    for x in ds[name] :
        dictionnary = json.loads(x)
        for y in dictionnary :
            if not (y.get("name") in genres) :
                genres.append(y.get("name"))
                count.append(1)
            else :
                count[genres.index(y.get("name"))]+=1
    return genres,count

def plotJson(datas,name,sizeX) :
    # Vizualisation of the popularity of a given column
    # sizeX is the size of the x_axis plotted

    genres,count = get_json(datas,name)
    index_sort = np.argsort(count)

    genres = np.flip(np.array(genres)[index_sort],axis=0)
    count = np.flip(np.array(count)[index_sort],axis=0)
    y_axis = count
    x_axis = [i for i in range(len(count))]
    x_label = genres
    plt.xticks(rotation=90, fontsize = 15)

    plt.ylabel("Number of occurences")
    if ( len(x_axis) > 25 ) :
        # It doesn't look great if the len is > to 25
        plt.bar(x_axis[:25], y_axis[:25], align = 'center', color='b')
        plt.xticks(x_axis[:25], x_label[:25])
    else :
        plt.bar(x_axis, y_axis, align = 'center', color='b')
        plt.xticks(x_axis, x_label)
        plt.xlabel(name)
    plt.title(name + " popularity")
    plt.show()

#plotJson(data,'genres',15)
#plotJson(data,'spoken_languages',0.05)

### Dealing with the occurences for genres
### The aim here is to keep the N most relevant values and to make the other as a "garbage" value

def group_by_occurences(ds,name,n) :
    # It splits the column "name" in n different features
    # It keeps the n-1 most important ones, and makes the others as another feature
    # It does it only for the json files
    # It returns the n names of the columns

    # Gives the different types and the occurences of each
    genres,count = get_json(ds,name)
    # Gives the indexes sorted of the list
    indexes = np.argsort(count)
    # We want to have the maximum, not the minimum : so we inverse the result
    indexes = [indexes[len(indexes)-i-1] for i in range(len(indexes))    ]
    # We get the result of the sort
    sorted_genres = np.array(genres)[indexes]
    sorted_count = np.array(count)[indexes]
    # We take the n-1 column values
    result = sorted_genres[:n-1]
    result = np.append(result,name+"_others")
    return result


def return_indexes_json(ds,name,category) :
    # Gives the indexes of istances where the json category appears
    genres = []
    # It will give all the different category of the column name
    for index,x in enumerate(ds[name])  :
        # Convert json file into dictionnary
        dictionnary = json.loads(x)
        for y in dictionnary :
            if(y.get("name") == category ) :
                genres.append(index)
    return genres



def create_columns_by_occurences(ds,name,n) :
    # It adds n columns to ds
    # It converts the n columns by adding 1-of-K encoding

    X = ds.copy()
    column_usable = group_by_occurences(ds,name,n)
    new_vals = np.zeros( (len(column_usable),), dtype=int )
    new_cols = np.append(X.columns.tolist(), column_usable)

    for x in column_usable :
        (n,p) = X.shape
        initial = [0 for i in range(n)]
        X.insert(p,x,initial,True)

    X[column_usable] = new_vals
    indexes_fullfill = []
    for i in range(len(column_usable)-1) :
        # Makes 1 where the values should be
        x = column_usable[i]
        indexes = return_indexes_json(ds,name,x)
        indexes_fullfill = indexes_fullfill + indexes
        X[x][indexes] = 1

    col = X[column_usable[len(column_usable)-1]]

    X[column_usable[len(column_usable)-1]] =  1
    X[column_usable[len(column_usable)-1]][indexes_fullfill] = 0
    X =X.drop(name,axis=1)

    return X

# 1-of-K encoding for the genres. 50 is a value superior to take into account all the values of genre
data = create_columns_by_occurences(data,"genres",50)

#data = create_columns_by_occurences(data,"spoken_languages",5)
# We first added spoken_languages by keeping the 5 most occured languages
# Then we realised that this feature isn't so relevant (few words in a given movie will be taken into account
# in spoken_languages). Indeed, it doesn't seem important


# We need to take into account the fact that a vote with a lot of vote_count is more accurate
# Thus, we need to find a way to create a new column to take this into account

# By using the Chebyshev's inequality, we can get that the probability of the expected vote of a film to be greater than
# vote_average - 10/sqrt(vote_count) is 0,99 (if we consider that the variance of vote is equal to 1)

data['grade'] = data.vote_average - 10/np.sqrt(data.vote_count)

# Plots the ratio rate
#plt.plot(data['budget'], data['revenue'], '+',color='r')
#plt.ylabel('Revenue')
#plt.xlabel('Budget')
#plt.title('Ratio graph')
#plt.show()

# Plots grande and vote_average depending on budget
#plt.plot(data.budget, data.vote_average,color='r', marker='+',linestyle='None',label='Vote average')
#plt.plot(data.budget, data.grade,color='b', marker='+',linestyle='None',label='Grade')
#plt.xlabel('budget')
#plt.legend()
#plt.title("Differences between vote average and grade depending on the budget")
#plt.show()


### Create a new column ratio to take into account both budget and revenue

data['ratio'] = data.budget / data.revenue
data.ratio[:5]


### We sort columns by grade and we drop vote, vote_count and popularity because they are no longer relevant
data.sort_values(by=['grade'],ascending = False)
data = data.drop(['vote_average','vote_count','popularity'],axis = 1)



# We use data about oscar nominees to give an "oscar_score" to each film,
# depending on the number of people in the cast that have already been nominee for an oscar.

oscars = pd.read_csv("../Data/Oscar/the_oscar_award.csv")
oscars = oscars[oscars['year_film'] > 1970]
oscarCount = oscars.groupby("name").agg("count").reset_index()[['name', 'film']].rename(columns={'film': 'count'}).sort_values(by = 'count', ascending = False)
oscar_name_string = oscarCount.name.to_string(index=False).lower()

for index, row in data.iterrows():
    oscar_nominee = 0
    oscar_score = 0

    nominations=[]
    dic = json.loads(row.cast)
    for ele in dic:
        nom = ele.get('name').lower()
        if (oscar_name_string.find(nom) > -1):
            oscar_nominee += 1
    if (oscar_nominee > 0):
        for s in range(1,oscar_nominee+1):
            oscar_score = oscar_score + (1/2)**s
        oscar_score = oscar_score*100
    data.loc[index, 'oscar_score'] = oscar_score


def sort_by_column(ds,columns) :
    # Returns a dataset sorted by the following column
    db = ds.copy().sort_values(by=columns)
    return db

# We need to take into account the time when the movies were released
data = sort_by_column(data, ['year', 'month'])


### Take care of production companies and production countries
### First of all, we vizualise the occurences of each in order to decide how we process it



### First of all, we will convert the string into dictonnary
### Then, we count the occurences in order to know what is most used
### Afterthat, we will only keep the first N ones which are most used


### Convert the column into dictionnary and count the occurences
def get_json(ds,name) :
    # Return a list with the different json file in the column name and the count of it
    genres = []
    count = []
    for x in ds[name] :
        dictionnary = json.loads(x)
        for y in dictionnary :
            if not (y.get("name") in genres) :
                genres.append(y.get("name"))
                count.append(1)
            else :
                count[genres.index(y.get("name"))]+=1
    return genres,count

def sup_to_value(liste1,liste2,value) :
    # Returns two lists where every elements of the second list is higher than "value"
    liste_result1,liste_result2 = [],[]
    for index,x in enumerate(liste1) :
        if liste2[index] >= value :
            liste_result1.append(x)
            liste_result2.append(liste2[index])
    return liste_result1,liste_result2

def plotJsonLimited(ds,name,sizeX,value_min) :
    # Vizualisation of the popularity of a given column
    # sizeX is the size of the x_axis plotted

    genres,count = get_json(ds,name)
    x_axis,y_axis = sup_to_value(genres,count, value_min)
    plt.xticks(rotation=90, fontsize = 15)
    plt.ylabel("Number of occurences")
    plt.xlabel(name)
    plt.bar(x_axis, y_axis, align = 'center', color='b')
    plt.title(name + " popularity")
    plt.show()


def get_N_first(ds,name,N) :
    # Gives the N first name and occurences of a given category coded in json
    genres,count = get_json(ds,name)
    value_min = np.flip(np.sort(count),axis=0)[N]
    types, count = sup_to_value(genres,count, value_min)
    count_indexes = np.argsort(count)
    types = np.flip(np.array(types)[count_indexes],axis=0)
    count = np.flip(np.array(count)[count_indexes],axis=0)
    return types,count

def get_N_first_all(ds,name) :
    # Gives the occurences of a given category coded in json
    genres,count = get_json(ds,name)
    value_min = np.flip(np.sort(count),axis=0)[-1]
    types, count = sup_to_value(genres,count, value_min)
    count_indexes = np.argsort(count)
    types = np.flip(np.array(types)[count_indexes],axis=0)
    count = np.flip(np.array(count)[count_indexes],axis=0)
    return types,count

def plot_N_first(ds,name,N) :
    # Plots the N first name and occurences of a given category coded in json
    # Nevertheless, it returns all the occurences (not the N first)
    types,count = get_N_first_all(ds,name)
    plt.xticks(rotation=90, fontsize = 15)
    plt.ylabel("Number of occurences")
    plt.xlabel(name)
    plt.bar(types[:N],count[:N],  align = 'center', color='b')
    plt.title(name + " popularity")
    #plt.show()
    return types,count

#plot_N_first(data, "production_countries",20)
types_p_companies, count_t_companies = plot_N_first(data, "production_companies",20)

#types_cast, count_cast = plot_N_first(data, "cast",20)
#types_crew, count_crew = plot_N_first(data, "crew",20)
### Here we do the 1-of-K encoding for the 5 most important production_countries
data = create_columns_by_occurences(data,"production_countries",5)

### We want to compute 1-of-K encoding for the production_companies
### Nevertheless, we will before categorize it in 3 categories
### To do that, we take into account the previous plot and we split by 3
### In this effect, when the occurences are > 85 or between 85 and 20 and <20


def get_index_json(x,liste_companies) :
    # Gives the instance indexes where the companies are
    print(x)
    dictionnary = json.loads(x)
    liste_index = []
    for y in dictionnary :
        name = y.get("name")
        for i,x in enumerate(liste_companies) :
            if name in x :
                liste_index.append(i)
                break
    if liste_index == [] :
        return 0
    return np.amax(liste_index)

def split_name_prod_comp(ds) :
    # Splits data into categories for the production companies
    high_comp_indexes = np.where(count_t_companies > 85)[0]
    high_comp = types_p_companies[high_comp_indexes]
    medium_comp_indexes = np.where( (count_t_companies <= 85) & (count_t_companies > 45) )[0]
    medium_comp = types_p_companies[medium_comp_indexes]
    low_comp_indexes = np.where(count_t_companies <= 45)[0]
    low_comp = types_p_companies[low_comp_indexes]
    all_companies = [low_comp,medium_comp,high_comp]
    print(ds.production_companies)
    ds.production_companies = ds.production_companies.apply(get_index_json, liste_companies = all_companies)
    ds = encode_language(ds.copy(),"production_companies",'prod_company')
    return ds

#data = split_name_prod_comp(data)
#print(data.head(2))

data_lang = data.groupby(['original_language']).size().reset_index(name='Occurences')
data_lang.sort_values(by='Occurences',ascending=False).plot(kind='bar',\
                                                      color='b', label='Revenue', grid=True, linestyle='-' )
#plt.ylabel("Occurences")
#plt.xlabel("Original languages")
#plt.title('Occurences of original languages')


def update_original_language(ds) :
    # Keeps the most important language
    # Fills the others with the value "others"
    ds.original_language = ds.original_language.map({'en' : 'en', 'fr' : 'fr', 'es' : 'es', 'de' :'de' ,'zh' : 'zh'})
    ds.original_language = ds.original_language.fillna('others')
    return ds

def encode_language(ds,name,prefix_name) :
    # 1-of-K encoding for name
    new_col = pd.get_dummies(ds[name],prefix = prefix_name)
    ds_conc = pd.concat([ds, new_col], axis=1)
    return ds_conc


data = update_original_language(data)
data = encode_language(data,'original_language','language')

### We will encode the runtime into different categories

def encode_runtime(ds,name) :

    ds[name].loc[ ds[name] < 60 ] = 1
    ds[name].loc[ (ds[name] >= 60) & (ds[name] < 90) ] = 2
    ds[name].loc[ (ds[name] >= 90) & (ds[name] < 120) ] = 3
    ds[name].loc[ ds[name] >= 120 ] = 4
    liste_time = ["less than 60 minutes","between 60 minutes and 90 minutes", \
                 "between 90 minutes and 120 minutes", "more than 120 minutes"]
    number_time = "Number of time of movies that last : "
    #for i in range(4) :
        #print(number_time + liste_time[i])
        #print(len(ds[name][ds[name] == i+1]))
    return ds

data = encode_runtime(data,"runtime")

### 1-of-K encoding for the runtime

def encoding(ds,name) :

    new_col = pd.get_dummies(ds[name])
    new_col.rename(columns = {1.0 : 'short_time',2.0:'medium_time',3.0:'quite_long_time',\
                             4.0:'long_time'}, inplace=True)
    ds_conc = pd.concat([ds, new_col], axis=1)
    return ds_conc

data = encoding(data,"runtime")
def remove_unused_data(ds,liste_columns) :
    # Removes all the features unused
    for x in liste_columns :
        ds = ds.drop(x, axis=1)
    return ds

unused = ['homepage','original_title','overview','runtime','status','tagline','title_x' ,\
          'title_y','original_language','production_companies',\
         'level_0','index','id','release_date','spoken_languages','month']
data = remove_unused_data(data,unused)
