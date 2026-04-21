import pickle

model = pickle.load(open("loan_model.pkl","rb"))
print(model.coef_.shape)