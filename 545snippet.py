
## 2.1 Load and Proprocess data [20 Points]
In this section, we will use this [properties dataset](https://www.kaggle.com/datasets/dataregress/dubai-properties-dataset/data) from Kaggle to explore linear regression, interpret its results, and perform coefficients hypothesis testings. Our goal is to model the relationship between the price of a house and all of its other features.

In supervised machine learning, regression and classification are two foundational tasks. While classification focuses on predicting discrete labels or categories, regression is concerned with predicting continuous values. One of the simplest and most widely used techniques for regression is linear regression, which models the relationship between input features and a continuous target variable by fitting a linear equation to the data.

To load the data:
1. Go to this [Kaggle link](https://www.kaggle.com) and create a Kaggle account (unless you already have one)
2. Go to Settings, scroll down to the "API" section and click on "Create New Token" to get the API key in the form of a json file `kaggle.json`
3. Upload the `kaggle.json` file to the default location in your Google Drive (Please **DO NOT** upload the json file into any _specific_ folder as it will be difficult for us to debug issues if you deviate from these instructions!).

This can be helpful for your project if you decide to use Kaggle for your final project or for future projects!
"""

# Run this cell to mount your drive (you will be prompted to sign in)
from google.colab import drive
drive.mount('/content/drive')

# Create the kaggle directory and
# (NOTE: Do NOT run this cell more than once unless restarting kernel)
!mkdir ~/.kaggle

# Read the uploaded kaggle.json file
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/

# Download dataset
!kaggle datasets download -d dataregress/dubai-properties-dataset

# Unzip folder in Colab content folder
!unzip /content/dubai-properties-dataset.zip

# TODO: Read the csv file and save it to a dataframe called "properties_df"
properties_df = pd.read_csv('properties_data.csv')

# Check out the first five rows
properties_df.head()

# Look at the data types in properties_df
properties_df.info()

"""### 2.1.1 Encode [5 points]
For linear regression to run, it only accepts numerical values, so we need to convert all non-numerical columns into numerical ones. All of the non-numerical columns we want to consider are booleans, so we can encode them by casting the type of their columns to integers.

Task:
- Save a copy of the current table as `encoded_properties_df`
- Drop the columns `id, price_per_sqft, latitude, longitude, neighborhood, and quality` from `encoded_properties_df`. We are doing this because `id`, `latitude`, and `longitude` are unlikely to have numerical significance, `quality` and `price_per_sqft` are too similar to our target variable, and there are too many categories for `neighborhood`
- Create a list of all categorical features (boolean type) and save it as `categorical_features`
- Create a list of all numerical features (non-boolean type) and save it as `numerical_features` to be used later. Do not include the `price` column in this list, as it will be our target variable
- Cast all categorical feature columns to integer type to encode them as either 0 or 1
"""

# DO NOT EDIT
encoded_properties_df = properties_df.copy()

# TODO: drop columns from `df_properties`
encoded_properties_df = encoded_properties_df.drop(columns=['id', 'price_per_sqft', 'latitude', 'longitude', 'neighborhood', 'quality'])

# TODO: create a list of all categorical features
print(encoded_properties_df.dtypes)
print("HELOPPP")
categorical_features = encoded_properties_df.columns.tolist()
for feature in categorical_features[:]:
    if encoded_properties_df[feature].dtype != 'bool':
        categorical_features.remove(feature)
        print(feature)

print(categorical_features)
# TODO: create a list of all numerical features
numerical_features = ['size_in_sqft', 'no_of_bedrooms', 'no_of_bathrooms']

# TODO: cast all categorical features to integer type

for feature in categorical_features:
    encoded_properties_df[feature] = encoded_properties_df[feature].astype(int)

# self-check: no need to change datatypes for this section
encoded_properties_df.info()

# Grader Cell (5 points)
grader.grade(test_case_id = 'properties_encode', answer = (categorical_features, numerical_features, encoded_properties_df))

"""### 2.1.2 Train-Test-Split [2 points]

The overall goal of this regression tasks is the use the independent variables we have to make a prediction on the price of the property, so the `price` column will be our target<br>


Conduct a train-test split on `encoded_properties_df`, assigning **80%** of the data to the training set and the remaining **20%** to the testing set. This ensures that the encoding and scaling we perform later are fitted only on the training data, preventing any spillover from the test data.

- Name the outputs as `X_train`, `X_test`, `y_train`, and `y_test`.
- Pass the argument `random_state = seed` in the function to fix the random state, ensuring consistency in our results.

The documentation of train_test_split can be found: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

# DO NOT EDIT
seed = 42

# TODO: perform train_test_split on encoded_properties_df
from sklearn.model_selection import train_test_split
selected_features = categorical_features + numerical_features

X_train, X_test, y_train, y_test = train_test_split(encoded_properties_df[selected_features],
    encoded_properties_df['price'],
    test_size=0.2,
    train_size=0.8,
    random_state = seed
)
print(X_train)

# Grader Cell (2 points)
grader.grade(test_case_id = 'properties_train_test_split', answer = (X_train, X_test, y_train, y_test))

"""### 2.1.3 Scaling [3 points]

As the final step in data pre-processing, we will prepare a scaled version of the data. Scaling refers to the process of mapping your features to a new range of values. This often helps machine learning models learn and converge faster. Some machine learning models are not *scale invariant*, meaning their ability to learn from the data can be impacted by the scale of the features. For example, models might give more influence to features with larger scales, implying that these features are more important than others, even when they should be treated with similar importance. Scaling features helps mitigate this issue.

There are several strategies for scaling, but in this section we will use **Standardization** on our continuous numerical features. Standardizing the data ensures that each feature is centered around zero ($\mu=0$) and has unit variance ($\sigma^2=1$).

**Task:**

* Apply standardization to the `numerical_features` in both the training and testing datasets using `StandardScaler` from sklearn
  * Make sure to include only the original numerical columns and not the newly encoded ones
* Store the results in `X_train_scaled` and `X_test_scaled`

**Note:**

* Prevent data leakage by ensuring that scaling parameters are learned only from the training set.
* While we only transform the original numerical columns, `X_train_scaled` and `X_test_scaled` should have the same shape as `X_train_encoded_full` and `X_test_encoded_full` from the end of section 2.1.2.

The documentation for StandardScaler can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
"""

# DO NOT EDIT
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# TODO: instantiate a StandardScaler object
scaler = StandardScaler()
# TODO: fit and transform on relevant columns from the training data
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# TODO: transform relevant columns from the testing data
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

"""Print the head of your df to see if the transformed columns' values make sense"""

X_train_scaled.head()

# Grader Cell (3 points)
grader.grade(test_case_id = 'properties_scaling', answer = (X_train_scaled, X_test_scaled))

"""### 2.1.4 Unregularized [5 points]

Use the `LinearRegression` class ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)) in scikit-learn to perform Linear Regression. Initialize a Linear regression model named `lr` with default parameters, fit the model to the training set, and then make predictions on the testing set.

Save your test predictions in an array named `y_pred_lr`, and report your [R-squared score](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score) for both train and test predictions
"""

from sklearn.linear_model import LinearRegression

# TO-DO: Initialize model with default parameters and fit it on the training set

# TO-DO: Use the model to predict on the test set and save these predictions as `y_pred_lr`

# TO-DO: Find the R-squared score for test and train and store the value in `lr_score_test` and `lr_score_train`
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_score_train = lr.score(X_train, y_train)
lr_score_test = lr.score(X_test, y_test)

# DO NOT EDIT
print(f"R² value for training set: {lr_score_train:.3f}")
print(f"R² value for test set: {lr_score_test:.3f}")

# Grader Cell (5 points)
grader.grade(test_case_id = 'properties_lr_unreg', answer = (lr_score_train, lr_score_test))

"""### 2.1.5 Ridge [5 points]
Use the `Ridge` class ([documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html))in scikit-learn to perform $L_2$ Regularized Linear Regression. Initialize a Ridge regression model named `lr_ridge` with regularization strength `alpha = 10`, fit the model to the training set, and then make predictions on both the training and testing set.

Save your test predictions in an array named `y_pred_ridge`, and report your [R-squared score](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.score) for both train and test predictions.

**Note:**
- Recall that Ridge regression is not scale-invariant, so you should use the scaled data from section 2.1.3
"""

from sklearn.linear_model import Ridge

# TO-DO: Initialize model and fit it on the training set

# TO-DO: Use the model to predict on the test set and save these predictions as `y_pred_ridge`

# TO-DO: Find the R-squared score for test and  and store the value in `ridge_score_train` and `ridge_score_test`
ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
ridge_score_train = ridge.score(X_train_scaled, y_train)
ridge_score_test = ridge.score(X_test_scaled, y_test)

### DO NOT EDIT
print(f"R² value for training set: {ridge_score_train:.3f}")
print(f"R² value for test set: {ridge_score_test:.3f}")

# Grader Cell (5 points)
grader.grade(test_case_id = 'properties_lr_ridge', answer = (ridge_score_train, ridge_score_test))

"""Before moving on to the next section, let's quickly reflect on the regularized versus unregularized linear regression models and think about some relevant questions (*note this exercise is NOT graded, just for your own exercise*):

1. When would we prefer L1 (LASSO) instead of L2? Recall that L1 (LASSO) versus L2 (Ridge) regularization: L1 is for sparisty (e.g., when you have more features than data points) and L2 is to encode an assumption that no input has drastically larger impact than another.
2. What do you notice about the $R^2$ scores of the Ridge and unregularized Linear Regression above? If Ridge is supposed to "improve" Linear Regression, we may find it unusual if that is not happening here. Under what circumstances would we consider using Ridge over unregularized Linear Regression
3. Would you expect different results if we made alpha bigger ( 𝛼→∞ )? smaller ( 𝛼→0 )? Why or why not?

## 2.2 Descriptive Model [10 Points]

Different from predictive model, for descriptive model we DO NOT perform train-test-split. The model is run on the whole set of data availbale.

### 2.2.1 Pre-process data for descriptive model [3 points]

Recall that in Section 2.1.3, scaling is trained only on the training set. However, since the descriptive model uses the entire dataset, scaling should be trained on the ***full*** dataset. Since we already encoded the entire dataset, we can reuse `encoded_properties_df` as our starting point.


**Task**:
  - Save the feature and target columns into `X` and `y` respectively. The target column is `price` and the feature columns are all of the other columns
  - Apply standardization to the original numerical columns using StandardScaler from sklearn
  - Make sure to include only the original numerical columns and not the newly encoded ones
"""

# Create feature and target tables
X = encoded_properties_df.drop(columns=['price'])
y = encoded_properties_df['price']

# TODO: instantiate a StandardScaler object

# TODO: fit and transform on relevant columns from the training data
scarler = StandardScaler()
X[numerical_features] = scarler.fit_transform(X[numerical_features])

# Grader Cell (3 points)
grader.grade(test_case_id = 'properties_descriptive_scaling', answer = (X, y))

"""### 2.2.2 Linear Regression [2 points]

Use the `LinearRegression` class ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)) in scikit-learn to perform Linear Regression. Initialize a Linear regression model named `lr_descriptive` with default parameters, fit the model to `X` and `y`, and report your [R-squared score](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score) between `X` and `y`
"""

from sklearn.linear_model import LinearRegression

# TODO: Initialize model with default parameters and fit it on the training set

# TODO: Find the R-squared score for test and  and store the value in `lr_score_descriptive`
lr = LinearRegression().fit(X, y)

lr_score_descriptive = lr.score(X, y)

# Grader Cell (2 points)
grader.grade(test_case_id = 'properties_lr_descriptive', answer = (lr_score_descriptive))

"""### 2.2.3 Interpreting Coefficients [5 points] (manually graded)
**Task**
1. Print out the coefficients of the descriptive linear regression model from Section 2.2.3 along with the feature names in a DataFrame:

| feature | coefficient |
|---------|----------|
|    ...     |    ...      |
|    ...     |    ...      |

2. Explain how size in square feet and the presence of a balcony affect the price

**Note**
1. Answer this question based on the coefficients of the model (e.g., if feature x increases by 1, how does that affect the cost?). Think about whether phrases like "balcony increases by 1" make sense, given it can only be 0 or 1; consider what an increase of 1 means for the scaled columns. Be explicit in explaining its meaning.
2. Reference of getting the coefficients from linear regression models: https://www.statology.org/sklearn-regression-coefficients/
3. Consider how you might you interpret the coefficients of your scaled features vs. your encoded features differently
"""

# TODO 1: print models' coefficients with matching column names
df = pd.DataFrame(zip(X.columns, lr.coef_))
print(df)

print("An increase in one standard deviation in the presense of a balcony increases the price from "
 + "the average number of properties with balconies, on average, increases the property value by by 1.424954e+05 dollars. "
 + "Though there can only be either a presence of a balcony, we don't know the exact standard deviation, so we cannot directly conclude"
  + " the average increase with the presense of a balcony with the coefficients provided in our scaled regression. That is something"
  +" we'd be able to conclude with the encoded regression instead. Instead, we can only conclude from the scaled regression that"
  +" an increase in one std out is positive, and is around 10k, which means that the presense of a balcony usually increases the price.")

"""**TODO 2: Explain how `size_in_sqft` and `balcony` affect the price** <br>

An increase in one standard deviation in the presense of a balcony increases the price from the average number of properties with balconies, on average, increases the property value by by 1.424954e+05 dollars. Though there can only be either a presence of a balcony, we don't know the exact standard deviation, so we cannot directly conclude the average increase with the presense of a balcony with the coefficients provided in our scaled regression. That is something we'd be able to conclude with the encoded regression instead. Instead, we can only conclude from the scaled regression that an increase in one std out is positive, and is around 10k, which means that the presense of a balcony usually increases the price.

Symmetrically, the same argument can be made for the size in sq_ft. An increase
in one standard deviation from the mean amount of size gives us an increased price of 2.816150e+06, though we cannot directly conclude how much change one square foot makes to the price.

Between the two, though, we can conclude that the size makes more of a difference to the price than the presense of a balcony, as the difference in price is much larger with the same standard deviation away from the average.

## 2.3 Hypothesis Testing [10 Pointes]

### 2.3.1 Testing for *all* coefficients = 0 [5 points] (1 manually graded)


Null Hypothesis: All coefficients of the unregularized model are 0, i.e., $\beta_0 = \beta_1 = ... = \beta_n = 0$.

**Task:**  

Use the code template given, report the following:

1. `original_r2`: The original model's $R^2$ value (you can use the value reported for the training data from Section 2.2.2).
2. `simulated_r2`: An array of $R^2$ values from the permuted data linear regression simulations.
3. `all_zero_p_value`: The p-value for the given null hypothesis, calculated by the proportion of times the simulated $R^2$ is larger than the original $R^2$.

**Note:**
- Use the method covered in lecture to complete this section (comparing $R^2$ values between the original regression and simulations of regressions on permuted data).
- Do **NOT** use any additional imports.
- We will manually check the code used and deduct points if necessary.
- Start from `X` and `y`
- Make a copy of `X` before shuffling
"""

from sklearn.utils import resample
from sklearn.metrics import r2_score

original_r2 = lr_score_descriptive
simulated_r2 = []

original_lr = LinearRegression().fit(
    encoded_properties_df.drop(columns=['price']).to_numpy(),
    encoded_properties_df['price'].to_numpy()
)
original_r2 = original_lr.score(
    encoded_properties_df.drop(columns=['price']).to_numpy(),
    encoded_properties_df['price'].to_numpy()
)

for _ in range(1000):
    # Shuffle y, not X
    y_shuffled = np.random.permutation(encoded_properties_df['price'].to_numpy(copy=True))

    # Fit the model and store the R²
    lr_shuffle = LinearRegression().fit(
        encoded_properties_df.drop(columns=['price']).to_numpy(), y_shuffled
    )
    simulated_r2.append(lr_shuffle.score(
        encoded_properties_df.drop(columns=['price']).to_numpy(), y_shuffled
    ))

simulated_r2 = np.array(simulated_r2)

# Calculate p-value correctly
all_zero_p_value = np.mean(simulated_r2 >= original_r2)

all_zero_p_value = float(np.mean(simulated_r2 > original_r2))
all_zero_p_value

# Grader Cell (4 points)
grader.grade(test_case_id = 'properties_all_zero', answer = (simulated_r2, all_zero_p_value, original_r2))

"""**TODO [Manually graded: 1 point]**

Q: should we reject the null hypothesis

A: yes, reject the null hypothesis

### 2.3.2  Testing for one particular coefficient is zero [5 points] (1 manually graded)
Null hypothesis: The coefficient of `private_jacuzzi` in the full unregularized linear regression is 0.

**Task:**  

Use the code template given, report the following:

1. `observed_r2`: The $R^2$ value when the linear regression is done without `priavte_jacuzzi` feature.
2. `simulated_r2_one`: An array of $R^2$ values from residual testing linear regressions
3. `all_zero_p_value`: The p-value for the given null hypothesis, calculated by the proportion of times the simulated $R^2$ is larger than the original $R^2$.

**Note:**
- Use the residual testing method covered in lecture to complete this section.
- Do **NOT** use any additional imports.
- We will manually check the code used and deduct points if necessary.
- start from `X` and `y`
"""

from sklearn.metrics import r2_score

# TODO: linear regression without the jacuzzi feature

modified_X = X.drop(columns=['private_jacuzzi'])
# TODO: find observed r2 score and initialize array to store simulated r2 scores

# Fit the original model
original_lr = LinearRegression().fit(modified_X, y)
observed_r2 = original_lr.score(modified_X, y)

# Get residuals
y_pred = original_lr.predict(modified_X)
residuals = y - y_pred

# Initialize an array to store simulated R² scores
simulated_r2 = []

for _ in range(1000):
    # Shuffle the residuals
    shuffled_residuals = np.random.permutation(residuals)

    # Create new y_shuffled by adding shuffled residuals back to predictions
    y_shuffled = y_pred + shuffled_residuals

    # Fit model on modified_X with shuffled y
    lr_shuffle = LinearRegression().fit(modified_X, y_shuffled)

    # Store the R² score
    simulated_r2.append(lr_shuffle.score(modified_X, y_shuffled))

# Convert list to array
simulated_r2_one = np.array(simulated_r2)

# Calculate the p-value
one_zero_p_value = np.mean(simulated_r2_one >= observed_r2)

print("P-value:", one_zero_p_value)

# Grader Cell (4 points)
grader.grade(test_case_id = 'properties_one_zero', answer = (simulated_r2_one, one_zero_p_value, observed_r2))

"""**TODO [Manually graded: 1 point]**

Q: should we reject the null hypothesis

A: no, cannot reject the null hypothesis"

# Part 3: Imbalanced Classification and Hyperparameter Tuning [42 Points]

### 3.1: Load Data
For this part, we will be using the Bank Marketing prediction dataset from Kaggle. This dataset contains 20 columns and about 40K rows.

We will start by loading the data. First, we will handle rows with "unknown" values in any of their fields, since "unknown" does not give us much useful information. Most columns have sufficiently few unknown values that we can simply drop any rows that have it, but the `default` column has over 21% "unknown", with nearly every other row being "no". This column most likely does not provide us with much predictive power and we can drop it.

In this section, we perform the following tasks for you. Please run all cells in this section before proceeding to the next.
*   Load the dataset called `bank-direct-marketing-campaigns.csv` using `pd.read_csv()`
*   Drop the column `default`
*   Remove all rows that have value 'unknown' in any of the columns `job`, `marital`, `education`, `housing`, `loan`
*   Reset index


**There is nothing to do in this section -- but you must run all cells before proceeding.**
"""

!kaggle datasets download ruthgn/bank-marketing-data-set

![ -f bank-direct-marketing-campaigns.csv ] || unzip bank-marketing-data-set.zip

bank_df = pd.read_csv('bank-direct-marketing-campaigns.csv')
bank_df.head()

# Get an overview of the data
bank_df.info()

bank_df = bank_df.drop(columns=['default'])

unknown_cols = ['job', 'marital', 'education', 'housing', 'loan']
for col in unknown_cols:
    bank_df = bank_df[bank_df[col] != 'unknown']

bank_df = bank_df.reset_index(drop=True)
bank_df

"""## 3.2: Data Pre-Processing & Feature Engineering [17 Points]

In a typical machine learning project, it's crucial to carefully process and analyze your data to understand the features you're working with. However, for this homework, the focus will be more on modeling rather than Exploratory Data Analysis (EDA). We've provided a dataset that is nearly ready for use, so you won't need to spend much time on EDA here.

However, this doesn’t mean you should skip EDA in your own projects! EDA will be an essential part of your project deliverables.

### 3.2.1: Encoding Categorical Features [9 Points]
Looking at our column types, we see that we have some features of type object, some of type int64, and some of type float64. Let's start by separating these.

**Task:**
*   Create two lists containing the column names of `numerical` and `categorical` features named `numerical_features` and `categorical_features` respectively
*   Sort these lists alphabetically

**Hint:**
* Consider using `.select_dtypes` from Pandas.

**Note:**
* Though `y` is not a feature (it is our target), please include it within one of the lists you create. We will address this later when we create our test-train split.
"""

# TODO: Populate the following lists
numerical_features = bank_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = sorted(numerical_features)
categorical_features = bank_df.select_dtypes(include=['object']).columns.tolist()
categorical_features = sorted(categorical_features)
print(numerical_features)
print(categorical_features)

# DO NOT CHANGE ----------------------------------------------------------------
print(f'There are {len(categorical_features)} categorical variables')
print(f'There are {len(numerical_features)} numerical variables')
# DO NOT CHANGE ----------------------------------------------------------------

# Grader Cell (2 points)
grader.grade(test_case_id='num_cat_features', answer=(numerical_features, categorical_features))

"""Now, let's focus on those categorical features and do some **encoding**.

Encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.

In this first part, we want to cast the columns containing binary variables (no/yes) into integer values (0 and 1).

**Task:**

*   You should create a copy of `bank_df` and save it as `binary_bank_df`, use this new data frame for this problem.
*   Find all columns that have 2 unique values (these are binary features).
*   Encode the columns with binary features using `OneHotEncoder` from sklearn **(refer to sklearn documentation for special arguments to the function)**.
*   Save results in `binary_bank_df`.

**Hint:**

* `binary_bank_df` should have the same dimensions/shape before and after encoding, and **the order of the columns should remain the same.**
* If not receiving full points, consider swapping the encoding scheme (swapping which values get the 0s and 1s)
"""

# TODO: Create a copy of bank_df and store it in binary_bank_df
binary_bank_df = bank_df.copy()

# TODO: For every column in bank_df, see if it has 2 unique values and if so, save to the binary_cols list
binary_cols = []
for col in bank_df.columns:
  if bank_df[col].nunique() == 2:
    binary_cols.append(col)

# Grader Cell (1 point)
grader.grade(test_case_id='binary_features', answer=binary_cols)



# TODO: Create
binary_bank_df = bank_df.copy()
print(binary_cols)
from sklearn.preprocessing import OneHotEncoder
print("Binary columns:", binary_cols)

binary_encoder = OneHotEncoder(drop='first', sparse_output=False)

binary_encoded = binary_encoder.fit_transform(binary_bank_df[binary_cols])

binary_encoded_df = pd.DataFrame(binary_encoded, columns=binary_encoder.get_feature_names_out(binary_cols))

binary_bank_df[binary_cols] = binary_encoded_df

display(binary_bank_df)
