```
madels = {
    "Logistic Regression" : LogisticRegression()
}

for i in range(len(list(models))):
    model = list(models.values()[i]
    model.fit(X_train,Y_train)

    Y_test_pred = model.predict(X_test)

    print("Accuracy "+str(Y_test_pred))

    print("-"*35)
    print("\n")

```