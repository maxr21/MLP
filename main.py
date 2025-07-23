import MLP
import Trainer

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

# desired outputs
ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP.MLP(3, [4, 4, 1])

trainer = Trainer.Trainer(mlp, xs, ys)

trainer.train(50, 0.1)

print([mlp(x) for x in xs])