#Model 4: Uses only bank information to determine credit risk

#Importing panda module for databases
import pandas as pd

#Importing the data from the dataset file
database = pd.read_csv("creditRiskDataset.csv")

#Filling all empty/unknown values from the dataset
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')

#Setting prediction rules
def predictRisk(row):
    predictedRiskScore = 0
    recordCheckings = str(database.iloc[row,6])
    recordSavings = str(database.iloc[row,5])

    if recordCheckings == "little":
        predictedRiskScore += 1
    elif recordCheckings == "rich":
        predictedRiskScore -= 1
    else:
        predictedRiskScore += 0
    
    if recordSavings == "little":
        predictedRiskScore += 1
    elif recordSavings == "quite rich":
        predictedRiskScore -= 1
    elif recordSavings == "rich":
        predictedRiskScore -= 1
    else:
        predictedRiskScore += 0
    
    if predictedRiskScore < 0:
        return("good",0)
    elif predictedRiskScore > 0:
        return("bad",0)
    else:
        return("bad",1)
    

timesUnsure = 0
riskPredictions = []
actualRisk = []
for i in range(900, 1119):
    prediction, isUnsure = predictRisk(i)

    timesUnsure += isUnsure
    actualRisk.append(str(database.iloc[i,10]))
    riskPredictions.append(prediction)


#Calculating accuracy
tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(riskPredictions)):
    if riskPredictions[i] == actualRisk[i]:
        if riskPredictions[i] == "good":
            tp += 1
        else:
            tn += 1
    else:
        if riskPredictions[i] == "good":
            fp += 1
        else:
            fn += 1

print("Total predictions: " + str(len(riskPredictions)))
print("True Positives: " + str(tp))
print("True Negatives: " + str(tn))
print("False Positives: " + str(fp))
print("False Negatives: " + str(fn))
print("Times Unsure: " + str(timesUnsure))

accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
recall = (tp / (tp + fn)) * 100
precision = (tp / (tp + fp)) * 100
f1Score = (2*((precision * recall)/(precision + recall)))

print("-=-=- MODEL ACCURACY STATISTICS-=-=-")
print("Accuracy = " + str(accuracy) + "%")
print("Recall = " + str(recall) + "%")
print("Precision = " + str(precision) + "%")
print("F1 Score = " + str(f1Score) + "%")