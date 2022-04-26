import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1a - Membuat dataframe dengan variabel yang ditentukan
# csv sudah memiliki header, langsung diload saja
df = pd.read_csv("insurance-cost.csv")
print(df.head())

print("")

# 1b - Mengisi missing values, cari variabel mana yang kosong
for i in df :
  print(i, " : ", df[i].isna().values.any())

print("")

# bmi: continuous, null diisi mean sesuai jenis kelaminnya.
df['bmi'] = df['bmi'].fillna(df.groupby(['sex'])['bmi'].transform('mean'))

# print contoh missing values di variabel bmi
print(df.loc[[30]])
print("")
print(df.loc[[51]])

print("")

# smoker: categorical, null diisi modus sesuai jenis kelaminnya.
df['smoker'] = df['smoker'].fillna(df.groupby(['sex'])['smoker'].transform(lambda x: pd.Series.mode(x)[0]))

# print contoh missing values di variabel smoker
print(df.loc[[32]])
print("")
print(df.loc[[53]])

print("")

#pembuktian sudah tidak ada nilai yang kosong
for i in df :
  print(i, " : ", df[i].isna().values.any())

print("")

#1c buat visualisasi terhadap setiap independent variable 
newData = df[["age","sex","bmi","children","smoker","region", "charges"]]

#sex, smoker, dan region adalah variable categorical, maka diubah menjadi angka terlebih dahulu
newData["sex"] = newData["sex"].map({'female': 0, 'male': 1})
newData["smoker"] = newData["smoker"].map({'no': 0, 'yes': 1})
newData["region"] = newData["region"].map({'southwest': 0, 'southeast': 1, 'northwest':2, 'northeast':3})

#print new data
print(newData.head())
print("")

#print histogram
newData.hist()
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()

#print relasi terhadap setiap independent variable menggunakan scatter plot
#X adalah age
plt.subplot(531)
plt.scatter(newData["age"], newData["sex"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("sex")

plt.subplot(532)
plt.scatter(newData["age"], newData["bmi"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("bmi")

plt.subplot(533)
plt.scatter(newData["age"], newData["children"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("children")

plt.subplot(534)
plt.scatter(newData["age"], newData["smoker"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("smoker")

plt.subplot(535)
plt.scatter(newData["age"], newData["region"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("region")

#x adalah sex
plt.subplot(536)
plt.scatter(newData["sex"], newData["bmi"], color="blue", alpha=0.25)
plt.xlabel("sex")
plt.ylabel("bmi")

plt.subplot(537)
plt.scatter(newData["sex"], newData["children"], color="blue", alpha=0.25)
plt.xlabel("sex")
plt.ylabel("children")

plt.subplot(538)
plt.scatter(newData["sex"], newData["smoker"], color="blue", alpha=0.25)
plt.xlabel("sex")
plt.ylabel("smoker")

plt.subplot(539)
plt.scatter(newData["sex"], newData["region"], color="blue", alpha=0.25)
plt.xlabel("sex")
plt.ylabel("region")

#X adalah bmi
plt.subplot(5,3,10)
plt.scatter(newData["bmi"], newData["children"], color="blue", alpha=0.25)
plt.xlabel("bmi")
plt.ylabel("children")

plt.subplot(5,3,11)
plt.scatter(newData["bmi"], newData["smoker"], color="blue", alpha=0.25)
plt.xlabel("bmi")
plt.ylabel("smoker")

plt.subplot(5,3,12)
plt.scatter(newData["bmi"], newData["region"], color="blue", alpha=0.25)
plt.xlabel("bmi")
plt.ylabel("region")

#X adalah children
plt.subplot(5,3,13)
plt.scatter(newData["children"], newData["smoker"], color="blue", alpha=0.25)
plt.xlabel("children")
plt.ylabel("smoker")

plt.subplot(5,3,14)
plt.scatter(newData["children"], newData["region"], color="blue", alpha=0.25)
plt.xlabel("children")
plt.ylabel("region")

#X adalah smoker
plt.subplot(5,3,15)
plt.scatter(newData["smoker"], newData["region"], color="blue", alpha=0.25)
plt.xlabel("smoker")
plt.ylabel("region")

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()

#1d Buat grafik correlation antara setiap independent variable dengan dependent variable
#scatter plot
plt.subplot(321)
plt.scatter(newData["age"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("age")
plt.ylabel("charges")

plt.subplot(322)
plt.scatter(newData["sex"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("sex")
plt.ylabel("charges")

plt.subplot(323)
plt.scatter(newData["bmi"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("bmi")
plt.ylabel("charges")

plt.subplot(324)
plt.scatter(newData["children"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("children")
plt.ylabel("charges")

plt.subplot(325)
plt.scatter(newData["smoker"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("smoker")
plt.ylabel("charges")

plt.subplot(326)
plt.scatter(newData["region"], newData["charges"], color="blue", alpha=0.25)
plt.xlabel("region")
plt.ylabel("charges")

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.rcParams["figure.figsize"] = [16,9]
plt.show()

#correlation matrixnya
corr_matrix = newData.corr()
sb.heatmap(data = corr_matrix, annot = True)
plt.show()

#2 Buatlah Training Set dan Test Set dengan proporsi 4:1
X = newData[["age", "bmi", "children", "smoker"]]
Y = newData["charges"]
XTrain, XTest, YTrain, YTest = train_test_split(X,Y, test_size = 0.2)

#print hasil training set yang sudah dipilih secara random
jointTrainData = pd.concat([XTrain, YTrain], axis=1)

pd.set_option('display.max_rows', None)  
print(jointTrainData)
pd.reset_option('display.max_rows')

print("Total Train Data: ", len(jointTrainData))
print("")

#3a Lakukan prediksi menggunakan test set
regression = LinearRegression().fit(XTrain, YTrain)

YPredict = regression.predict(XTest)

#3b Tampilkan hasil prediksi tersebut dan nilai sebenarnya dari test set
for i in range(len(XTest)):
  print("Predicted : %-15.5lf" % (YPredict[i]), " Actual : %-15.5lf" % (YTest.values[i]))

print("Total Test Set: ", len(XTest))
print("")

#4a Tampilkan persamaan regresi
print("Intercept (Konstanta)    : ", regression.intercept_)
print("Kofisien  (age, bmi, children, smoker) : ", regression.coef_)
print("Persamaan  Regresi: ", \
  regression.intercept_ , "+", \
  regression.coef_[0], "x1 + ", \
  regression.coef_[1], "x2 + ", \
  regression.coef_[2], "x3 + ", \
  regression.coef_[3], "x4"
)
print("")

#4c Evaluasi dari hasil nilai prediksi
print("Mean Absolute Error (MAE) : ", mean_absolute_error(YTest, YPredict))
print("Mean Squared Error (MSE) : ", mean_squared_error(YTest, YPredict))
print("Root Mean Squared Error (RMSE) : ",  mean_squared_error(YTest, YPredict, squared=False))
print("R2 Score   : ", r2_score(YTest, YPredict))
print("")

#5 plot yang menggambarkan hasil dari predicted value dan actual value
plotNewData = pd.DataFrame({"Actual": YTest, "Predicted": YPredict})
sb.regplot(x='Actual',y='Predicted', data=plotNewData, fit_reg=True, scatter_kws={'color': 'darkred', 'alpha': 0.3, 's': 100})

plt.show()