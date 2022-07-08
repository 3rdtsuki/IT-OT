import torch

cnn = torch.load('cnn_minist.pkl')

test_output = cnn(test_x[:20])
pred_y = torch.max(test_output, 1)[1].data.numpy()

# info = torch.max(test_output,1)[1]
# print(test_output)
# print(info)

print(pred_y, 'prediction number')
print(test_y[:20].numpy(), 'real number')


test_output1 = cnn(test_x)
pred_y1 = torch.max(test_output1, 1)[1].data.numpy()
accuracy = float((pred_y1 == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print('accuracy',accuracy)