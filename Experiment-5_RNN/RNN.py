import torch
import torch.nn as nn

text = """
Artificial intelligence is changing the world.
Machine learning helps computers learn from data.
Deep learning uses neural networks to solve complex problems.
Recurrent neural networks process sequential information.
They remember information from previous steps using hidden states.
Natural language processing helps machines understand human language.
Text generation predicts the next character based on previous characters.
Learning requires data, patience, and repeated experiments.
Technology can help people solve important problems.
Researchers continuously improve intelligent systems.
""" * 20

chars= sorted(set(text))
char_to_id = {ch: i for i , ch in enumerate(chars)}
id_to_char = {i: ch for i , ch in enumerate(chars)}

vocab_len = len(chars)
print(vocab_len)
print(chars)

encoded = torch.tensor([char_to_id[ch] for ch in text])
print(encoded[:20])

seq_length = 40

X = [] #Input
Y = [] #output

for i in range(0, len(encoded) - seq_length):
    X.append(encoded[i : i + seq_length])
    Y.append(encoded[i + seq_length])
X = torch.stack(X)
Y = torch.stack(Y)
print(X[0])
print(Y[0])

X = nn.functional.one_hot(X, num_classes=vocab_len).float()
Y = nn.functional.one_hot(Y, num_classes=vocab_len).float()

class RNNText(nn.Module):
  def __init__(self, vocab_len, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(vocab_len, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, vocab_len)

  def forward(self, X):
    out, _ =self.rnn(X)
    out = self.fc(out)
    return out

model = RNNText(vocab_len, 128)  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

target_indices = torch.argmax(Y, dim=1).long() if Y.dim() > 1 else Y.long()
for epoch in range(20):
  prediction= model(X)

  loss = criterion(prediction[:, -1, :], target_indices)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print(f"Epoch: {epoch}, Loss: {loss.item()}")

def generate_text(model, seed, length=200, temperature=0.8):
    model.eval()
    generated = seed

    with torch.no_grad():
        for _ in range(length):

            context = generated[-seq_length:]
            idx = torch.tensor([[char_to_id[ch] for ch in context]])
            x = nn.functional.one_hot(idx, num_classes=vocab_len).float()

            output = model(x)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1).item()
            generated += id_to_char[next_id]

    return generated
seed = "Machine learning helps computers learn "
print(generate_text(model, seed, length=300))
