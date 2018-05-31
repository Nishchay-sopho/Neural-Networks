import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import matplotlib.lines as mlines

df= pd.read_csv('/home/nishchay/Documents/Arcon/Day7/winequality-red.csv')
X1=df.iloc[:,11].values
Y1=df.iloc[:,0].values
Y2=df.iloc[:,1].values

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 20)
cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
ax1.grid(True)
plt.title('Wine Quality Correlation')
labels=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide','asdf']
ax1.set_xticklabels(labels,fontsize=6)
ax1.set_yticklabels(labels,fontsize=6)
fig.colorbar(cax, ticks=[- .6,- .5,- .4,- .3,- .2,- .1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
plt.show()
#################################################################################
col_labels = df.columns[1:]
corMat2 = df.corr().values[::-1]
fig, axes = plt.subplots(nrows=1,ncols=1)
ax0 = axes
ax0.set_xticks(np.linspace(.5,12.5,11))
ax0.set_xticklabels(col_labels,rotation=45)

ax0.set_yticks(np.linspace(.5,12.5,11))
ax0.set_yticklabels(col_labels[::-1],rotation=45)
#ax0.set_yticklabels(col_labels,rotation=45)

#visualize correlations using heatmap
cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
cmap = cm.get_cmap('jet', 20)
fig.colorbar(cax, ticks=[- .6,- .5,- .4,- .3,- .2,- .1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
plt.pcolor(corMat2,cmap='jet')
plt.show()

############################################################################

plt.plot(Y1,X1,'r--',Y2,X1,'bs')
plt.xlabel('Wine Quality')
plt.ylabel('fixed acidity')
red_line = mlines.Line2D(Y1,X1,color='red',marker='_',markersize=10,label='Fixed Acidity')
blue_line=mlines.Line2D(Y2,X1,color='blue',marker='|',markersize=10,label='Volatile Acidity')
plt.legend(handles=[red_line,blue_line])
plt.show()


labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
