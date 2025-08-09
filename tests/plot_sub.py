import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load results
df = pd.read_csv('experiment_metrics.csv')

# Choose which metric to plot
metric = 'eig_direct_forward_error'  # or any other column from df

# Unique z values (sorted)
z_values = sorted(df['z'].unique())

# Plot heatmaps for each z
for z in z_values:
    # Filter for fixed z
    df_z = df[df['z'] == z]
    
    # Pivot into Lx x Ly grid
    pivot_table = df_z.pivot(index='Ly', columns='Lx', values=metric)
    
    # Sort axes so plots are ordered
    pivot_table = pivot_table.sort_index(ascending=True)  # Ly
    pivot_table = pivot_table[pivot_table.columns.sort_values()]  # Lx
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        cmap='viridis',
        annot=False,
        cbar_kws={'label': metric}
    )
    plt.title(f"{metric} heatmap at z = {z:.3f}")
    plt.xlabel("Lx")
    plt.ylabel("Ly")
    plt.tight_layout()
    plt.show()

# Optional: Trend plots vs. z for a fixed Lx,Ly
fixed_Lx = df['Lx'].iloc[0]
fixed_Ly = df['Ly'].iloc[0]
df_fixed = df[(df['Lx'] == fixed_Lx) & (df['Ly'] == fixed_Ly)]

plt.figure(figsize=(8, 4))
plt.plot(df_fixed['z'], df_fixed[metric], marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("z (propagation distance)")
plt.ylabel(metric)
plt.title(f"{metric} vs z at Lx={fixed_Lx}, Ly={fixed_Ly}")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

