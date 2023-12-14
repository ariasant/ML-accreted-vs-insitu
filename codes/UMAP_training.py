import numpy as np
import pickle
import umap
from umap.umap_ import nearest_neighbors

file =  np.load('/mnt/data1/users/ariasant/data/train_ds_umap.npz')
data = file['data']
labels = file['labels']

# Pre-calculate k-nearest-neighbours to data points to speed up computation
knn = nearest_neighbors(data,
                        n_neighbors=100,
                        metric='cosine',
                        metric_kwds=None,
                        angular=False,
                        random_state=42)

print('KNN calculated')

# Define UMAP model
umap_model = umap.UMAP(random_state=42,
                       low_memory=True,
                       verbose=True,
                       precomputed_knn=knn,
                       init='random'
)

# Learn embedding to maximise separation between accreted and in-situ
X_umap =  umap_model.fit_transform(data,
                                   y=labels
)

print('UMAP model trained')

# Save model
f_name = 'UMAP_model_umap_ds.sav'
pickle.dump(umap_model, open(f_name, 'wb'))
print('Model saved')

# Save projection of training data
np.save('X_umap_train_umap_ds', X_umap)
print('Training data projection saved')

# Apply model to test data
test_file = np.load('/mnt/data1/users/ariasant/data/test_ds.npz')
test_data = test_file['data']
print('Data loaded.')

X_test = umap_model.transform(test_data)
print('Test data projected.')
# Save projected data
np.save('X_umap_test_umap_ds', X_test)
print('Test data projections saved')

