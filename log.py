import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG: provide your log file paths ===
log_files = {
    'DCA': 'logs/log_dca.csv',
    'CTrans': 'logs/log_ctrans_early.csv',
    'Baseline': 'logs/log_baseline.csv'
}

# === Function to extract detailed metrics ===
def extract_metrics(path, model_name):
    df = pd.read_csv(path)
    best_row = df.loc[df['val_iou'].idxmax()]
    return {
        'Model': model_name,
        'Best Epoch': int(best_row['epoch']),
        'Best val_iou': best_row['val_iou'],
        'Best val_dice': best_row['val_dice'],
        'Final val_iou': df.iloc[-1]['val_iou'],
        'Final val_dice': df.iloc[-1]['val_dice'],
        'Min val_loss': df['val_loss'].min(),
        'Avg val_iou': df['val_iou'].mean(),
        'Std val_iou': df['val_iou'].std(),
        'Avg val_dice': df['val_dice'].mean(),
        'Std val_dice': df['val_dice'].std(),
        'Total Time (min)': df['epoch_time'].sum() / 60,
        'Total Epochs Run': len(df)
    }

# === Process all logs ===
results = []
dfs = {}
for model_name, path in log_files.items():
    metrics = extract_metrics(path, model_name)
    results.append(metrics)
    dfs[model_name] = pd.read_csv(path)  # store df for plotting

# === Create comparison dataframe ===
comparison_df = pd.DataFrame(results)

# === Display summary ===
print("=== Segmentation Model Comprehensive Comparison ===")
print(comparison_df)

# === Save summary to CSV for records/reports ===
comparison_df.to_csv("model_comparison_summary.csv", index=False)

# === Plotting function with best epoch markers ===
def plot_metric(metric_name, ylabel, title):
    plt.figure(figsize=(12,5))
    for model_name, df in dfs.items():
        plt.plot(df['epoch'], df[metric_name], label=f'{model_name} {metric_name}')
        best_epoch = df.loc[df[metric_name].idxmax()]['epoch']
        best_value = df[metric_name].max()
        plt.scatter(best_epoch, best_value, marker='o', s=50)
        plt.text(best_epoch, best_value, f"{best_value:.3f}", fontsize=9)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# === Plot val_iou, val_dice, val_loss ===
plot_metric('val_iou', 'Validation IoU', 'Validation IoU Progression per Model')
plot_metric('val_dice', 'Validation Dice', 'Validation Dice Progression per Model')
plot_metric('val_loss', 'Validation Loss', 'Validation Loss Progression per Model')
