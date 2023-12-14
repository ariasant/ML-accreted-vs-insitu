import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


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
         'right' : True,
         'major.size' : 7,
         'minor.size' : 4,
         'major.width' : 0.5,
         'minor.width' : 0.35,
         'direction' : 'in',
         'minor.visible' : True,
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



# Load data from test datasets
test_halos =  [29,30,34,42]
df = pd.concat([pd.read_pickle('/mnt/data1/users/ariasant/data/dataframes/G{:02}_df.pkl'.format(halo)) for halo in test_halos])

# Pre-process datasets as done for training the ml models
df = df[df['r_cylindrical']<=np.sqrt(2)*50]
df = df[np.abs(df['z_kpc'])<50]
df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]

# Calculate useful quantities
df['v'] = np.sqrt(df['vx_kms']**2 + df['vy_kms']**2 + df['vz_kms']**2)
df['Lx_kpc_kms'] = df['y_kpc']*df['vz_kms'] - df['z_kpc']*df['vy_kms']
df['Ly_kpc_kms'] = - df['x_kpc']*df['vz_kms'] + df['z_kpc']*df['vx_kms']
df['Lz_kpc_kms'] = df['x_kpc']*df['vy_kms'] - df['y_kpc']*df['vx_kms']

# Create column to put label predictions (insitu/accreted)
df['predictions'] = np.ones(df.shape[0])

#====
# Remove background of in-situ stars using only
# z-cut like in arXiv:2208.01056v1
#====
idx_insitu = np.where(np.abs(df['z_kpc']) < 2.5)
df['predictions'].iloc[idx_insitu] = 0

np.save('/mnt/data1/users/ariasant/training_results/Z_cut_predictions',df['predictions'].values)
df['predictions'].iloc[:] = 1

#====
# Remove background of in-situ stars using only
# velocity-cut like in doi: 10.1038/s41586-018-0625-x,
# and arXiv:2201.02404 
#====
v_LOS = 232 #kms 
idx_insitu = np.where( (df['vtheta']-v_LOS)**2 + df['vsigma']**2 < 210**2)
df['predictions'].iloc[idx_insitu] = 0

np.save('/mnt/data1/users/ariasant/training_results/V_cut_predictions',df['predictions'].values)
df['predictions'].iloc[:] = 1

#====
# Select accreted stars using a selection cut in the kinematics 
# like in doi: 10.1051/0004-6361/201936135
#====
df['r_spherical'] = np.sqrt(df['x_kpc']**2 + df['y_kpc']**2 + df['z_kpc']**2)
df['v'] = np.sqrt(df['vx_kms']**2+df['vy_kms']**2+df['vz_kms']**2)
# define the orbital circularity parameter
df['epsilon'] = df['Lz_kpc_kms'] / (df['v']*df['r_spherical'])
# in-situ stars are stars within 3.5 kpc from the centre (bulge)
# or stars within 5 kpc from the plane of the disk and with a 
# circular orbit (within the disk)
idx_insitu = np.where(((df['r_spherical']<3.5) |
                       ( (np.abs(df['z_kpc'])<5) & (np.abs(df['epsilon'])>0.5) )))[0]
df['predictions'].iloc[idx_insitu] = 0

np.save('/mnt/data1/users/ariasant/training_results/KinematicCut_predictions',df['predictions'].values)
df['predictions'].iloc[:] = 1

#====
# Select accreted stars using a selection cut in FeH and vtheta 
# like in https://doi.org/10.1093/mnras/stx3262
#====

predictions = np.array([])

# Perform analysis on halo-by-halo basis 

# Plot FeH vs vtheta to see the cut-off between accreted and in-situ stars
def plot_FeH_vtheta(df,halo):

    fig, ax = plt.subplots(1,2, sharey=True, figsize=(8,4))

    fig.suptitle('G{:02}'.format(halo))

    v0=ax[0].hexbin(df['vtheta'].values,
                df['FeH'].values,
                gridsize=(300,519),
                extent=[-350,350,-4,1],
                norm=mpl.colors.LogNorm(),
                cmap='inferno'
                )
    ax[0].set_aspect(700/5)
    ax[0].set_ylabel(r'$[\mathrm{Fe}/\mathrm{H}]$')
    ax[0].set_xlabel(r'$v_{\theta} \, [\mathrm{km}\mathrm{s}^{-1}]$')

    cbar = fig.colorbar(v0,ax=ax[0], shrink=0.5)
    cbar.set_label('Number of Stars')

    v1=ax[1].hexbin(df['vtheta'].values,
                df['FeH'].values,
                gridsize=(300,519),
                extent=[-350,350,-4,1],
                C=df['insitu_flag'].values,
                cmap=mpl.colors.ListedColormap(['#00429d', '#73a2c6', '#ffffe0', '#f4777f', '#93003a'])
                )

    ax[1].set_aspect(700/5)
    ax[1].set_xlabel(r'$v_{\theta} \, [\mathrm{km}\mathrm{s}^{-1}]$')

    cbar = fig.colorbar(v1,ax=ax[1], shrink=0.5)
    cbar.set_label('Accreted Stars Fraction')

    fig.savefig('G{:02}_FeH_vtheta.png'.format(halo), dpi=400)

# Apply cut-off
for halo in test_halos:
    df = pd.read_pickle('/mnt/data1/users/ariasant/data/dataframes/G{:02}_df.pkl'.format(halo))

    df = df[df['r_cylindrical']<=np.sqrt(2)*50]
    df = df[np.abs(df['z_kpc'])<50]
    df = df[(df['FeH']>=-4) & (df['FeH']<1.5)]
    df = df[(df['aFe']<1.5) & (df['aFe']>-0.5)]

    plot_FeH_vtheta(df, halo)


predictions =  np.ones(df.shape[0])
idx_insitu = np.where( (df['FeH']>-0.5) & (df['vtheta']>150) )[0]
predictions[idx_insitu] = 0
    

np.save('/mnt/data1/users/ariasant/training_results/FeH_vtheta_predictions', predictions)
