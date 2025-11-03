#Model 1: Using only credit amount to determine credit risk

#Importing panda module for databases
import pandas as pd

#Importing the data from the dataset file
database = pd.read_csv("creditRiskDataset.csv")

#Filling all empty/unknown values from the dataset
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')


#Initializing arrays for classifying training data
highRiskCreditAmounts = []
lowRiskCreditAmounts = []

#Filling AI databanks from first 900 records for training, classifying credit amounts, durations etc etc into arrays based on if credit risk was good or bad (low or high)
for i in range(0,900):
    
    if (database.iloc[i,10] == "bad"):
        highRiskCreditAmounts.append(int(database.iloc[i,7]))
    
    elif (database.iloc[i,10] == "good"):
        lowRiskCreditAmounts.append(int(database.iloc[i,7]))

#Creating averages to check against to make predictions
highRiskCreditAmounts.sort()
averageHighRiskCreditAmount = sum(highRiskCreditAmounts) / len(highRiskCreditAmounts)
Q1HighRiskCreditAmount = highRiskCreditAmounts[int(len(highRiskCreditAmounts)/4)]
Q3HighRiskCreditAmount = highRiskCreditAmounts[(int(len(highRiskCreditAmounts)/4))*3]
lowestHighRiskCreditAmount = highRiskCreditAmounts[0]
print("Average High risk credit amount: " + str(averageHighRiskCreditAmount))

lowRiskCreditAmounts.sort()
averageLowRiskCreditAmount = sum(lowRiskCreditAmounts) / len(lowRiskCreditAmounts)
Q1LowRiskCreditAmount = lowRiskCreditAmounts[int(len(lowRiskCreditAmounts)/4)]
Q3LowRiskCreditAmount = lowRiskCreditAmounts[(int(len(lowRiskCreditAmounts)/4))*3]
highestLowRiskCreditAmount = lowRiskCreditAmounts[(len(lowRiskCreditAmounts)-1)]
print("Average Low risk credit amount: " + str(averageLowRiskCreditAmount))

#Setting thresholds for risk assessment
if lowestHighRiskCreditAmount > highestLowRiskCreditAmount:
    creditAmountThreshold = highestLowRiskCreditAmount + ((lowestHighRiskCreditAmount - highestLowRiskCreditAmount)/2)
    creditAmountConfidence = 3
elif Q1HighRiskCreditAmount > Q3LowRiskCreditAmount:
    creditAmountThreshold = Q3LowRiskCreditAmount + ((Q1HighRiskCreditAmount - Q3LowRiskCreditAmount)/2)
    creditAmountConfidence = 2
elif averageHighRiskCreditAmount > averageLowRiskCreditAmount:
    creditAmountThreshold = averageLowRiskCreditAmount + ((averageHighRiskCreditAmount - averageLowRiskCreditAmount)/2)
    creditAmountConfidence = 1
else:
    creditAmountThreshold = 0
    creditAmountConfidence = 0
print("Credit Amount Threshold = " + str(creditAmountThreshold) + " (Confidence: " + str(creditAmountConfidence) + ")")


#Setting prediction rules
def predictRisk(row):
    predictedRiskScore = 0
    recordCreditAmount = int(database.iloc[row,7])

    if recordCreditAmount > creditAmountThreshold:
        predictedRiskScore += creditAmountConfidence
    else:
        predictedRiskScore -= creditAmountConfidence

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
print("-=-=- MODEL ACCURACY: " + str(accuracy) + "% -=-=-")