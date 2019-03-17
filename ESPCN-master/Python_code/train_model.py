import torch

def train(epoch, model, criterion, optimizer, training_dataloader):
	global min_loss
	global position
	global update_min
	global training_size
	global device
	epoch_loss = 0
	# Calculate Loss
	for batch in training_dataloader:
		input, target = batch[0].to(device), batch[1].to(device)

		optimizer.zero_grad()
		loss = criterion(model(input), target)
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
	epoch_loss /= training_size

	# Update min_loss
	if epoch_loss <= min_loss:
		position, min_loss, update_min = epoch, epoch_loss, True

	# Print loss
	if epoch % 10 == 0:
		print ("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))

def Train_model(model, criterion, optimizer, training_dataloader, learning_rate, nEpochs, _device, model_save_path):
	global min_loss
	global position
	global update_min
	global training_size
	global device
	min_loss, position = 1, 0
	update_min = False
	device = _device
	training_size = len(training_dataloader)

	for epoch in range(1, nEpochs + 1):
		if epoch == 500:
			optimizer.lr = learning_rate / 10
		train(epoch, model, criterion, optimizer, training_dataloader)
		
		#checkpoint(epoch)
		model_out_path = "{}/epoch{}.pth".format(model_save_path, epoch)
		torch.save(model, model_out_path)

		if epoch in [nEpochs/10 * x for x in range(1, 11)]:
			if update_min:
				print ("===>> Epoch {} Complete, Min_loss Changed at epoch {}:".format(epoch, position))
				print ("===>> Min. Loss: {:4f}".format(min_loss))
				update_min = False
			else:
				print ("===>> Epoch {} Complete, Min_loss Remained at epoch {}:".format(epoch, position))
				print ("===>> Min. Loss: {:4f}".format(min_loss))

	return min_loss, position