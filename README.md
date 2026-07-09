A simple classification model loosely based off of the model in NNFS, using few external libraries

The model currently uses the Adam optimiser

It also has the ability to use dropout, but for this use case it tanks the models accuracy so it's temporarily removed

Dev Roadmap:
- [ ] Fashion-MNIST (Dense baseline)
- [ ] Document accuracy and speed
- [ ] Conv2D + Pool + Flatten
- [ ] Switch to flat vec
- [ ] GPU (wgpu)
- [ ] BatchNorm / LayerNorm
- [ ] Residual connections
- [ ] Fashion-MNIST v2 (CNN + GPU, compare to baseline)
- [ ] Self-attention from scratch
- [ ] Multi-head attention + positional encoding
- [ ] Minimal transformer block
