import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_logs(path: str, tag: str) -> pd.DataFrame:
	"""
	Loads scalar data from a TensorBoard log file.
	"""
	ea = event_accumulator.EventAccumulator(path,
		size_guidance={event_accumulator.SCALARS: 0})
	ea.Reload()

	if tag not in ea.Tags()['scalars']:
		print(f"Warning: Tag '{tag}' not found in {path}. Skipping.")
		return pd.DataFrame()

	scalar_events = ea.Scalars(tag)
	steps = [event.step for event in scalar_events]
	values = [event.value for event in scalar_events]

	return pd.DataFrame({'Epoch': steps, 'Value': values})

base_log_dir = './runs'
runs = {
	'deit': os.path.join(base_log_dir, 'deit'),
	'resnet32_finetune': os.path.join(base_log_dir, 'resnet32_finetune'),
	'resnet32_transfer': os.path.join(base_log_dir, 'resnet32_transfer'),
	'vit': os.path.join(base_log_dir, 'vit')
}
tags = {
	'Accuracy/train': ('Accuracy', 'Train'),
	'Accuracy/val': ('Accuracy', 'Validation'),
	'Loss/train': ('Loss', 'Train'),
	'Loss/val': ('Loss', 'Validation')
}

all_data_dfs = []
for model_name, path in runs.items():
	if not os.path.exists(path):
		print(f"Warning: Directory not found at {path}. Skipping model '{model_name}'.")
		continue
	for tag_name, (metric, split) in tags.items():
		df_run = load_tensorboard_logs(path, tag_name)
		if not df_run.empty:
			df_run['Model'] = model_name
			df_run['Metric'] = metric
			df_run['Split'] = split
			all_data_dfs.append(df_run)

if not all_data_dfs:
	print("Error: No data was loaded. Please check your paths and tag names.")
else:
	temp_df = pd.concat(all_data_dfs)
	
	if 'resnet32_transfer' in temp_df['Model'].unique():
		transfer_max_step = temp_df[temp_df['Model'] == 'resnet32_transfer']['Epoch'].max()
		print(f"Offset determined from resnet32_transfer's final step: {transfer_max_step}")

		for i, df in enumerate(all_data_dfs):
			if not df.empty and df['Model'].iloc[0] == 'resnet32_finetune':
				df['Epoch'] += transfer_max_step
				all_data_dfs[i] = df 
	else:
		print("Warning: 'resnet32_transfer' data not found. Cannot apply offset.")


	df_combined = pd.concat(all_data_dfs, ignore_index=True)

	sns.set_theme(style="darkgrid")

	g = sns.relplot(
		data=df_combined,
		x="Epoch",
		y="Value",
		hue="Model",
		col="Split",
		row="Metric",
		kind="line",
		height=4,
		aspect=1.2,
		linewidth=1.5,
		facet_kws={'sharey': False, 'sharex': True}
	)


	g.fig.suptitle("Model Metrics", y=1.03, fontsize=16)
	g.set_titles("{row_name} / {col_name}", size=12)
	g.set_axis_labels("Epoch", "Value", size=12)
	sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -0.08), ncol=4, title=None, frameon=False)
	
	plt.savefig("model_comparison_plot.png", dpi=300, bbox_inches='tight')
	plt.show()
