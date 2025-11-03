#Model 3: Using only duration to determine credit risk

#Importing panda module for databases
import pandas as pd

#Importing the data from the dataset file
database = pd.read_csv("creditRiskDataset.csv")

#Filling all empty/unknown values from the dataset
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')
database['Saving accounts'] = database['Saving accounts'].fillna('unknown')


#Initializing arrays for classifying training data
highRiskDurations = []
lowRiskDurations = []

#Filling AI databanks from first 900 records for training, classifying credit amounts, durations etc etc into arrays based on if credit risk was good or bad (low or high)
for i in range(0,900):
    
    if (database.iloc[i,10] == "bad"):
        highRiskDurations.append(int(database.iloc[i,8]))
    
    elif (database.iloc[i,10] == "good"):
        lowRiskDurations.append(int(database.iloc[i,8]))

#Creating averages to check against to make predictions
highRiskDurations.sort()
averageHighRiskDuration = sum(highRiskDurations) / len(highRiskDurations)
Q1HighRiskDuration = highRiskDurations[int(len(highRiskDurations)/4)]
Q3HighRiskDuration = highRiskDurations[(int(len(highRiskDurations)/4))*3]
lowestHighRiskDuration = highRiskDurations[0]
print("Average High risk duration: " + str(averageHighRiskDuration))

lowRiskDurations.sort()
averageLowRiskDuration = sum(lowRiskDurations) / len(lowRiskDurations)
Q1LowRiskDuration = lowRiskDurations[int(len(lowRiskDurations)/4)]
Q3LowRiskDuration = lowRiskDurations[(int(len(lowRiskDurations)/4))*3]
highestLowRiskDuration = lowRiskDurations[(len(lowRiskDurations)-1)]
print("Average low risk duration: " + str(averageLowRiskDuration))


#Setting thresholds for risk assessment
if lowestHighRiskDuration > highestLowRiskDuration:
    durationThreshold = highestLowRiskDuration + ((lowestHighRiskDuration - highestLowRiskDuration)/2)
    durationConfidence = 3
elif Q1HighRiskDuration > Q3LowRiskDuration:
    durationThreshold = Q3LowRiskDuration + ((Q1HighRiskDuration - Q3LowRiskDuration)/2)
    durationConfidence = 2
elif averageHighRiskDuration > averageLowRiskDuration:
    durationThreshold = averageLowRiskDuration + ((averageHighRiskDuration - averageLowRiskDuration)/2)
    durationConfidence = 1
else:
    durationThreshold = 0
    durationConfidence = 0
print("Duration Threshold = " + str(durationThreshold) + " (Confidence: " + str(durationConfidence) + ")")



#Setting prediction rules
def predictRisk(row):
    predictedRiskScore = 0
    recordDuration = int(database.iloc[row,8])

    if recordDuration > durationThreshold:
        predictedRiskScore += durationConfidence
    else:
        predictedRiskScore -= durationConfidence
    
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