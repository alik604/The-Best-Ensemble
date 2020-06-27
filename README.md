# The Best Ensemble

`The Best Ensemble` Python scikit-learn sample code to act as a very competitive baseline to beat in classification tasks

## Usage

```python

#foo()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


wanted_explained_variance_ratio = 0.99
steps_down = 2
wanted_n_components = X_train.shape[1]
first_time = True

for i in range(X_train.shape[1]-1, 1, -steps_down):
  total_var_ratio = round(np.sum(PCA(n_components=i).fit(X_train).explained_variance_ratio_),5)
  print('i =', i, 'with a variance ratio of', total_var_ratio)
  if total_var_ratio < wanted_explained_variance_ratio and first_time:
    wanted_n_components = i + steps_down
    first_time = False
print("We should set n_components to: ",wanted_n_components)

pca = PCA(n_components=wanted_n_components ) # want 26ish to 28ish components, if you want 99% of variance explained
_ = pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
ETC = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
XGB = xgboost.XGBClassifier(n_estimators=150, n_jobs=-1) # 0.806
GBM = lightgbm.LGBMClassifier(objective='multiclass', n_estimators= 500)

list_of_CLFs_names = []
list_of_CLFs = [DTC, RFC, ETC, XGB, GBM]
ranking = []

for clf in list_of_CLFs:
    _ = clf.fit(X_train,y_train)
    pred = clf.score(X_test,y_test)
    name = str(type(clf)).split(".")[-1][:-2]
    print("Acc: %0.5f for the %s" % (pred, name))

    ranking.append(pred)
    list_of_CLFs_names.append(name)


#CAUTION: Very Expensive
CBC = catboost.CatBoostClassifier(eval_metric='AUC',use_best_model=True,random_seed=42)
CBC.fit(X_train,y_train,eval_set=(X_test, y_test))
pred = CBC.score(X_test,y_test)
name = str(type(CBC)).split(".")[-1][:-2]
print("Acc: %0.5f for the %s" % (pred, name))

ranking.append(pred)
list_of_CLFs.append(CBC)
list_of_CLFs_names.append(name)


best = max(ranking)
avg = sum(ranking)/len(ranking)
variance = avg - best
to_remove = ranking - avg - variance
to_remove_alt = ranking - best - variance
print(list_of_CLFs_names)
print(to_remove)
print(to_remove_alt)
ranking = np.array(ranking)[to_remove > 0]
list_of_CLFs = np.array(list_of_CLFs)[to_remove > 0]


eclf = EnsembleVoteClassifier(clfs=list_of_CLFs, refit=False, voting='soft')
eclf.fit(X_train, y_train)
pred = eclf.score(X_test, y_test)
print("Acc: %0.5f for the %s" % (pred, str(type(eclf)).split(".")[-1][:-2]))


pred = eclf.predict(X_test)
probas = eclf.predict_proba(X_test)
skplt.metrics.plot_roc(y_true=y_test, y_probas=probas)
plt.show()

```

### Benchmark

```python

if type(y_test) is (pd.core.frame.DataFrame or pd.core.series.Series):
  y_test = y_test.values

def benchmark(y_test=y_test, pred=pred, pred_proba=None):
    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, auc, roc_curve, roc_auc_score, hamming_loss, precision_score, recall_score, f1_score, log_loss

    print("Hamming_loss: \t\t", round(hamming_loss(y_test, pred), 3))                       # HL=1-Accuracy
    print("Precision_score: \t", round(precision_score(y_test, pred, average='micro'), 3))  # tp / (tp + fp)
    print("Recall_score: \t\t", round(recall_score(y_test, pred, average='micro'), 3))      # tp / (tp + fn)
    print("F1 score: \t\t", round(f1_score(y_test, pred, average='micro'), 3))              # 2 * (precision * recall) / (precision + recall)
    print("------------------------------")
    print("Accuracy_score: \t", round(accuracy_score(y_test, pred), 3))

    fpr, tpr, _ = roc_curve(y_test == 6, pred == 6)                                         # 6 is "Normal", rest are 'bad'
    print("False Positive Rate - binarized ", fpr)
    print("True Positive Rate  - binarized ", tpr)
    print("Area of ROC: \t", round(auc(fpr, tpr), 3))                                       # Apx definite integral... of the ROC
    print("------------------------------")
    if pred_proba is not None:
      print("Log loss (categorical cross entropy): \t", round(log_loss(y_test, pred), 3))   # -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

benchmark(y_test, pred, probas)

print(multilabel_confusion_matrix(y_test,pred))
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[Unlicense](https://choosealicense.com/licenses/unlicense/)
