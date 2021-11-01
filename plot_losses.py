import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sns.set_context('poster')

save_dir1 = sys.argv[1]
save_dir2 = sys.argv[2]

losses = np.load(save_dir1+'/losses.npz')
losses_concat = np.load(save_dir2+'/losses.npz')

plt.figure(figsize=(10,5))
plt.semilogy(losses['training'].mean(1), '-', color='royalblue', label='gnblock3 train')
plt.semilogy(losses['validation'].mean(1), '-', color='indianred', label='gnblock3 val')

plt.semilogy(losses_concat['training'].mean(1), '--', color='royalblue', label='gnblock6 train')
plt.semilogy(losses_concat['validation'].mean(1), '--', color='indianred', label='gnblock6 val')

plt.legend()
plt.grid()
sns.despine()
plt.title(save_dir1.split('/')[-1])
plt.tight_layout()
plt.show()
