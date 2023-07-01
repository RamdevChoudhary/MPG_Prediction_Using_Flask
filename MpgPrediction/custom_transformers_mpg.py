from sklearn.base import BaseEstimator, TransformerMixin
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.in_column = "name"
    def fit(self, X, y=None):
        # X -> Data Frame
        return self
    def transform(self, X , y=None):
        X[self.in_column] = X[self.in_column].str.split(" ", expand=True)[0]
        return X
    def fit_transform(self, X, y=None):
        self.fit(X, y)# parameter
        return self.transform(X, y) # act to transform


# # Documentation for `FeatureAdder` Class

# ## Overview
# `FeatureAdder` is a custom transformer class designed to be used within a scikit-learn
#  pipeline. It inherits from `BaseEstimator` and `TransformerMixin`, allowing it to be
#  seamlessly integrated into a scikit-learn data preprocessing pipeline. The main purpose of 
# this transformer is to process a specific feature of the input DataFrame and modify it to extract
#  valuable information for downstream machine learning models.

# ## Class Attributes

# ### `in_column` (str)
# - Default value: "name"
# - Description: The name of the column in the input DataFrame that will be processed by the 
# `FeatureAdder` transformer. This column is expected to contain text data.

# ## Class Methods

# ### `fit(self, X, y=None)`
# - Parameters:
#   - `X` (pandas DataFrame): The input DataFrame from the pipeline.
#   - `y` (array-like, default=None): The target labels (optional, not used in this transformer).

# - Returns: `self`
# - Description: This method is called during the fitting process of the pipeline and does not
#  perform any significant calculations. It is included to maintain consistency within the scikit-learn pipeline structure.

# ### `transform(self, X, y=None)`
# - Parameters:
#   - `X` (pandas DataFrame): The input DataFrame from the pipeline.
#   - `y` (array-like, default=None): The target labels (optional, not used in this transformer).

# - Returns: Transformed `X` (pandas DataFrame)
# - Description: This method is responsible for processing the specified column in the input DataFrame.
#  It extracts the first part of the text in the specified column by splitting the 
# strings on spaces and taking the first part. The transformed DataFrame is then returned.

# ### `fit_transform(self, X, y=None)`
# - Parameters:
#   - `X` (pandas DataFrame): The input DataFrame from the pipeline.
#   - `y` (array-like, default=None): The target labels (optional, not used in this transformer).

# - Returns: Transformed `X` (pandas DataFrame)
# - Description: This method combines the `fit()` and `transform()` methods. It first calls the `fit()` method 
# (which does nothing), and then applies the `transform()` method to the input DataFrame `X`. The resulting 
# transformed DataFrame is returned.

# Example Usage

# ```python
# Create an instance of FeatureAdder
# feature_adder = FeatureAdder()

# Define the input DataFrame with the 'name' column containing text data
# data = pd.DataFrame({'name': ['Toyota Corolla', 'Ford Mustang', 'Honda Civic']})

# Transform the input DataFrame using the FeatureAdder
# transformed_data = feature_adder.transform(data)
# Print the transformed DataFrame
# print(transformed_data)

# Output:
#       name
# 0   Toyota
# 1     Ford
# 2    Honda
# ```

# In this example, the `FeatureAdder` transformer is applied to the 'name' column of the input DataFrame
#  `data`. It extracts the first word from each row of the 'name' column, resulting in the transformed 
# DataFrame `transformed_data`.
