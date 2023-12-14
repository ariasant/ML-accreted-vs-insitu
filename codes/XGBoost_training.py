import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

#Load data
train_file = np.load('/mnt/data1/users/ariasant/data/train_ds.npz', allow_pickle=True)

print('Data Loaded',flush=True)
train_data = train_file['data']
train_labels = train_file['labels']

print('Dataset size: {}'.format(train_data.shape[0]))

print('Training started',flush=True)
model = XGBClassifier(n_estimators=1000,
                      learning_rate=0.09,
                      max_depth=8,
                      n_jobs=-1
)
#Start training
model.fit(train_data,train_labels)

#Save model and training outputs
file_name = "xgb_model.pkl"
pickle.dump(model, open(file_name, "wb"))

# Make predictions
test_file = np.load('/mnt/data1/users/ariasant/data/test_ds_last.npz', allow_pickle=True)
test_data = test_file['data']
predictions = model.predict_proba(test_data)[:,1]

# Save predictions
np.save('test_pred_xgb_SN', predictions)

# Make predictions on VALIDATION set
test_file = np.load('/mnt/data1/users/ariasant/data/val_ds_last.npz', allow_pickle=True)
test_data = test_file['data']
predictions = model.predict_proba(test_data)[:,1]

# Save predictions
np.save('val_pred_xgb_last', predictions)


# Plot feature importance
import matplotlib.pyplot as plt
import matplotlib as mpl

#Adjust formatting for plots
font = {'family' : 'sans-serif',
'weight' : 'medium',
'size'   : 15,
'variant' : 'normal',
'style' : 'normal',
'stretch' : 'normal',
}

xtick = {'top' : True,
         'bottom' : True,
         'major.size' : 7,
         'minor.size' : 4,
         'major.width' : 0.5,
         'minor.width' : 0.35,
         'direction' : 'in',
         'minor.visible' : True,
         'color' : 'black',
         'labelcolor' : 'black'
         }

ytick = {'left' : True,
         'right' : False,
         'major.size' : 7,
         'minor.size' : 4,
         'major.width' : 0.5,
         'minor.width' : 0.35,
         'direction' : 'out',
         'minor.visible' : False,
         'color' : 'black',
         'labelcolor' : 'black'
         }

mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.figsize'] = (6.973848069738481, 4.310075139476229)
mpl.rcParams['figure.subplot.hspace'] = 0.01

mpl.rc('font', **font)
mpl.rc('xtick', **xtick)
mpl.rc('ytick', **ytick)
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams["font.sans-serif"] = ["DejaVu Serif"]
mpl.rcParams['mathtext.fontset']='dejavuserif'

importances = model.feature_importances_
features = [r'$r$', r'$z$', r'$v_{\theta}$', r'$v_{\sigma}$', r'[Fe/H]', r'[$\alpha$/Fe]', r'$M_{G}$', r'$BP-RP$']

fig,ax =  plt.subplots(layout='constrained')

y_pos = np.arange(1,len(importances)+1)

cmap = mpl.colormaps['tab10']
xgb_color = cmap.colors[4]
ax.barh(y_pos, width=importances, color=xgb_color, tick_label=features)

ax.set_yticks(y_pos)
ax.set_xlabel('Feature Importance')

fig.savefig('XGB_feature_importance.png',dpi=400)





