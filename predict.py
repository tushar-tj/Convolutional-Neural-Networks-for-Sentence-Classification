import argparse
import pickle
import torch

parser = argparse.ArgumentParser(description='CNN Based Text classifier CNN - Predict')
parser.add_argument('-model_path', type=str, required=True, help='path of the model to use for prediction')
parser.add_argument('-preprocessor_path', type=str, required=True, help='path of the preprocessor to use for prediction')
parser.add_argument('-sentence', type=str, required=True, help='Sentence to use for prediction')
args = parser.parse_args()

print ('Arguments')
print (args)

def predict(model, preprocessor, sentence):
    with open(preprocessor, 'rb') as f:
        data_preprocessor = pickle.load(f)

    model = torch.load(model)
    model.eval()

    x = data_preprocessor.sent2Index([sentence])
    y = model(x)

    pred = data_preprocessor.index2class[y.argmax(-1).item()]
    return pred

pred = predict(args.model_path, args.preprocessor_path, args.sentence)
print ('Sentence : {} | {}'.format(args.sentence, pred))