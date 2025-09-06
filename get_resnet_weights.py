import torch
import os

os.makedirs('checkpoints', exist_ok=True)

model_urls = {
	'resnet20': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th',
	'resnet32': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th',
}

for model_name, model_url in model_urls.items():
	local_path = f"checkpoints/{model_name}_cifar10.pth"
	state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device("cpu"))["state_dict"]

	new_state_dict = {}
	for k, v in state_dict.items():
		if 'linear' in k:
			k = k.replace('linear', 'fc')

		if k.startswith('module.'):
			# Remove the "module." prefix
			new_state_dict[k[7:]] = v
		else:
			new_state_dict[k] = v

	torch.save(new_state_dict, local_path)

	print(f"{model_name} CIFAR-10 weights saved to {local_path}")
