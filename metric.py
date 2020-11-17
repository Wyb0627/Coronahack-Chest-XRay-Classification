from sklearn import metrics
import pandas as pd

label_test = pd.read_csv(r'C:\Users\Yubo\Desktop\Python\CIS\test_label.csv')
ans = pd.read_csv(r'C:\Users\Yubo\Desktop\Python\CIS\ans.csv')
ans_list = []

for i in ans.index:
    label0 = ans.loc[i].values[0]
    label1 = ans.loc[i].values[1]
    label2 = ans.loc[i].values[2]
    if label0 > label1 and label0 > label2:
        ans_list.append(0)
    elif label1 > label0 and label1 > label2:
        ans_list.append(1)
    elif label2 > label1 and label2 > label0:
        ans_list.append(2)

print('precision_score:')
print(metrics.precision_score(label_test['Label'], ans_list, average="weighted"))
print('accuracy_score:')
print(metrics.accuracy_score(label_test['Label'], ans_list))
print('f1_score:')
print(metrics.f1_score(label_test['Label'], ans_list, average="weighted"))
print('recall_score:')
print(metrics.recall_score(label_test['Label'], ans_list, average="weighted"))
