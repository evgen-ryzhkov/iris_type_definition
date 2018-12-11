from scripts.data import IrisData
from scripts.model import IrisClassifier


# Step 1
# getting data and familiarity with them
iris_data_obj = IrisData()
iris_data = iris_data_obj.get_full_data_set()

print(iris_data.keys())
# iris_df = iris_data_obj.get_x_y_df()
# iris_data_obj.search_correlation(iris_df)
# iris_data_obj.show_data_distribution(iris_df)
X_train, y_train, X_validation, y_validation, X_test, y_test = iris_data_obj.get_split_data_set()


# Step 2
# training and testing model
iris_clf = IrisClassifier()
model_parameters = iris_clf.train_model(X_train, y_train)
iris_clf.check_model('Validation', X_validation, y_validation, model_parameters)
iris_clf.check_model('Test', X_test, y_test, model_parameters)