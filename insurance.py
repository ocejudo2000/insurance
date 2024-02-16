# load the dataset
from pycaret.datasets import get_data
insurance= get_data("insurance")

#init environment
from pycaret.regression import *
r1= setup(insurance, target = 'charges', session_id=123,
          normalize = True,
          polynomial_features = True, 
          bin_numeric_features = ['age', 'bmi'])

# train the model
lr = create_model('lr')

#save pipeline/model
save_model(lr, model_name = "deployment_28042020")
 
